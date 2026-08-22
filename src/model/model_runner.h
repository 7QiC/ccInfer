#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "backend/backend.h"
#include "base/execution.h"
#include "base/result.h"
#include "cache/kv_cache_manager.h"
#include "core/traits.h"
#include "model/config.h"
#include "model/model.h"

namespace ccinfer {

class ModelRunner {
public:
    template <typename Traits>
    static Result<std::vector<WorkItemResult>> inference(Model& model, const PhysicalBatch& batch,
                                                         Backend& backend, KVCacheManager& kv_mgr,
                                                         const SamplingParams& sampling = {}) {
        static_assert(runner_traits_valid_v<Traits>, "RunnerTraits has unknown dtype tags");

        // Only BF16 weights / activations / KV + FP32 logits.
        if constexpr (!std::is_same_v<typename Traits::WeightTag, ops::BFloat16Tag> ||
                      !std::is_same_v<typename Traits::KVTag, ops::BFloat16Tag> ||
                      !std::is_same_v<typename Traits::ActivationTag, ops::BFloat16Tag> ||
                      !std::is_same_v<typename Traits::LogitsTag, ops::Float32Tag>) {
            (void)Traits{};
            return std::unexpected(ErrorCode::Unsupported);
        }

        // Validate physical batch.
        if (!batch.token_ids.valid() || !batch.positions.valid() || !batch.slot_mapping.valid() ||
            !batch.block_table.valid() || !batch.query_start_loc.valid() ||
            !batch.context_lens.valid() || !batch.logits_indices.valid()) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        const int T = batch.num_tokens;
        const int B = batch.batch_size;
        if (T <= 0 || B <= 0 || batch.max_blocks_per_req <= 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (batch.item_indices.size() != static_cast<std::size_t>(B) ||
            batch.item_seq_ids.size() != static_cast<std::size_t>(B) ||
            batch.item_kinds.size() != static_cast<std::size_t>(B)) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (batch.mode != ForwardMode::Prefill && batch.mode != ForwardMode::Decode &&
            batch.mode != ForwardMode::Mixed) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
        if (batch.mode == ForwardMode::Decode && T != B) {
            return std::unexpected(ErrorCode::ModelShapeMismatch);
        }

        if (batch.mode != ForwardMode::Mixed) {
            const WorkKind expected = batch.mode == ForwardMode::Prefill ? WorkKind::PrefillChunk
                                                                         : WorkKind::DecodeOneToken;
            for (int i = 0; i < B; ++i) {
                if (batch.item_kinds[i] != expected)
                    return std::unexpected(ErrorCode::InvalidArgument);
                if (batch.item_indices[i] >
                    static_cast<std::size_t>(std::numeric_limits<int>::max()))
                    return std::unexpected(ErrorCode::InvalidArgument);
            }
        }

        // slot_mapping[t] and block_table entries are validated by BatchTranslator:
        //   0 <= slot_mapping[t] < kv_mgr.max_slots()
        //   block_table entries are valid block ids or -1.

        const auto& cfg = model.config();
        const int V = cfg.vocab_size_;
        if (V <= 0) return std::unexpected(ErrorCode::ModelConfigInvalid);

        // Validate sampling params.
        if (sampling.top_k < 0 || sampling.top_k > V) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
        if (!std::isfinite(sampling.top_p) || sampling.top_p <= 0.0f) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
        if (!std::isfinite(sampling.temperature) || sampling.temperature < 0.0f) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
        const bool greedy =
            (sampling.top_k == 0 && sampling.top_p >= 1.0f && sampling.temperature <= 0.0f);
        if (!greedy) {
            return std::unexpected(ErrorCode::Unsupported);
        }

        // BatchTranslator already built this per-item metadata; reuse it instead
        // of copying query_start_loc/logits_indices back from device.
        if (batch.item_token_counts.size() != static_cast<std::size_t>(B) ||
            batch.sample_flags.size() != static_cast<std::size_t>(B)) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
        for (int i = 0; i < B; ++i) {
            const int len = batch.item_token_counts[i];
            if (len <= 0) return std::unexpected(ErrorCode::InvalidArgument);
            if (batch.mode == ForwardMode::Decode && len != 1)
                return std::unexpected(ErrorCode::InvalidArgument);
            if (batch.item_kinds[i] == WorkKind::DecodeOneToken && len != 1)
                return std::unexpected(ErrorCode::InvalidArgument);
            if (batch.mode == ForwardMode::Decode && !batch.sample_flags[i])
                return std::unexpected(ErrorCode::InvalidArgument);
        }

        if (batch.max_position_id < 0) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        // Build ForwardInput.
        ForwardInput input;
        input.token_ids = batch.token_ids;
        input.positions = batch.positions;
        input.num_tokens_ = T;
        input.max_position_id_ = batch.max_position_id;
        input.kv_mgr_ = &kv_mgr;
        input.slot_mapping = batch.slot_mapping;
        input.block_table = batch.block_table;
        input.query_start_loc = batch.query_start_loc;
        input.context_lens = batch.context_lens;
        input.batch_size_ = B;
        input.max_blocks_per_req_ = batch.max_blocks_per_req;
        input.mode_ = batch.mode;

        constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
        const std::size_t T_size = static_cast<std::size_t>(T);
        const std::size_t V_size = static_cast<std::size_t>(V);
        if (V_size > kMax / T_size) return std::unexpected(ErrorCode::InvalidArgument);
        const std::size_t logits_elems = T_size * V_size;
        if (logits_elems > kMax / sizeof(float)) return std::unexpected(ErrorCode::InvalidArgument);
        auto logits_r = Tensor::empty(backend, ops::DType::kFloat32, {T, V});
        if (!logits_r) return std::unexpected(logits_r.error());
        auto tokens_r = Tensor::empty(backend, ops::DType::kInt32, {B});
        if (!tokens_r) return std::unexpected(tokens_r.error());

        ForwardOutput output;
        output.logits = std::move(*logits_r);
        output.tokens_out = std::move(*tokens_r);
        auto fwd_r = model.forward(input, output, backend);
        if (!fwd_r) return std::unexpected(fwd_r.error());

        auto s_r = ops::map_result(ops::greedy_sample(output.logits, batch.logits_indices,
                                                      &output.tokens_out, backend.context()));
        if (!s_r) return std::unexpected(s_r.error());

        // D2H: sampled tokens.
        std::vector<int32_t> tokens_host(B);
        {
            auto r = backend.memcpy_d2h(tokens_host.data(), output.tokens_out.data(),
                                        static_cast<std::size_t>(B) * sizeof(int32_t));
            if (!r) return std::unexpected(r.error());
        }

        std::vector<WorkItemResult> results;
        results.reserve(static_cast<std::size_t>(B));
        for (int i = 0; i < B; ++i) {
            WorkItemResult wr;
            wr.item_index = static_cast<int>(batch.item_indices[i]);
            wr.seq_id = batch.item_seq_ids[i];
            wr.kind = batch.item_kinds[i];
            wr.sampled_tokens = batch.sample_flags[i]
                                    ? std::vector<int32_t>{tokens_host[i]}
                                    : std::vector<int32_t>{};
            wr.tokens_consumed = batch.item_token_counts[i];
            wr.eos = false;
            results.push_back(std::move(wr));
        }

        return results;
    }
};

}  // namespace ccinfer
