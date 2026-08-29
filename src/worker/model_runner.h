#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "backend/backend.h"
#include "cache/block_storage.h"
#include "common/error_code.h"
#include "common/physical_batch.h"
#include "core/traits.h"
#include "model/config.h"
#include "model/model.h"

namespace ccinfer {

class ModelRunner {
public:
    template <typename Traits>
    static Result<std::vector<WorkItemResult>> inference(Model& model, const PhysicalBatch& batch,
                                                         Backend& backend,
                                                         BlockStorage& block_storage,
                                                         const SamplingParams& sampling = {}) {
        static_assert(runner_traits_valid_v<Traits>, "RunnerTraits has unknown dtype tags");

        // Only BF16 weights / activations / KV + FP32 logits.
        if constexpr (!std::is_same_v<typename Traits::WeightTag, ccop::BFloat16Tag> ||
                      !std::is_same_v<typename Traits::KVTag, ccop::BFloat16Tag> ||
                      !std::is_same_v<typename Traits::ActivationTag, ccop::BFloat16Tag> ||
                      !std::is_same_v<typename Traits::LogitsTag, ccop::Float32Tag>) {
            (void)Traits{};
            return std::unexpected(ErrorCode::Unsupported);
        }

        const int T = batch.num_tokens;
        const int B = batch.batch_size;
        assert(batch.token_ids.valid() && batch.positions.valid() && batch.slot_mapping.valid() &&
               batch.block_table.valid() && batch.query_start_loc.valid() &&
               batch.context_lens.valid() && batch.logits_indices.valid());
        assert(T > 0 && B > 0 && batch.max_blocks_per_req > 0);
        assert(batch.item_seq_ids.size() == static_cast<std::size_t>(B));
        assert(batch.item_kinds.size() == static_cast<std::size_t>(B));
        assert(batch.mode == ForwardMode::Prefill || batch.mode == ForwardMode::Decode ||
               batch.mode == ForwardMode::Mixed);
        assert(batch.mode != ForwardMode::Decode || T == B);
        assert(batch.item_token_counts.size() == static_cast<std::size_t>(B));
        assert(batch.sample_flags.size() == static_cast<std::size_t>(B));
        assert(batch.logits_rows_host.size() == static_cast<std::size_t>(batch.num_logits));
        assert(batch.max_position_id >= 0);

        const auto& cfg = model.config();
        const int V = cfg.vocab_size_;
        assert(V > 0);
        assert(sampling.top_k >= 0 && sampling.top_k <= V);
        assert(std::isfinite(sampling.top_p) && sampling.top_p > 0.0f);
        assert(std::isfinite(sampling.temperature) && sampling.temperature >= 0.0f);

        const bool greedy =
            (sampling.top_k == 0 && sampling.top_p >= 1.0f && sampling.temperature <= 0.0f);
        if (!greedy) {
            return std::unexpected(ErrorCode::Unsupported);
        }

        // Build ForwardInput.
        ForwardInput input;
        input.token_ids = batch.token_ids;
        input.positions = batch.positions;
        input.num_tokens_ = T;
        input.max_position_id_ = batch.max_position_id;
        input.block_storage_ = &block_storage;
        input.slot_mapping = batch.slot_mapping;
        input.block_table = batch.block_table;
        input.query_start_loc = batch.query_start_loc;
        input.context_lens = batch.context_lens;
        input.batch_size_ = B;
        input.max_blocks_per_req_ = batch.max_blocks_per_req;
        input.mode_ = batch.mode;
        input.logits_indices_host = batch.logits_rows_host;
        input.num_logits_ = batch.num_logits;

        constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
        const std::size_t V_size = static_cast<std::size_t>(V);
        const int num_logits = batch.num_logits > 0 ? batch.num_logits : 1;
        const std::size_t logits_rows = static_cast<std::size_t>(num_logits);
        assert(V_size <= kMax / logits_rows);
        const std::size_t logits_elems = logits_rows * V_size;
        assert(logits_elems <= kMax / sizeof(float));
        auto logits_r = Tensor::empty(backend, ccop::DType::kFloat32, {num_logits, V});
        if (!logits_r) return std::unexpected(logits_r.error());
        auto tokens_r = Tensor::empty(backend, ccop::DType::kInt32, {B});
        if (!tokens_r) return std::unexpected(tokens_r.error());

        ForwardOutput output;
        output.logits = std::move(*logits_r);
        output.tokens_out = std::move(*tokens_r);
        auto fwd_r = model.forward(input, output, backend);
        if (!fwd_r) return std::unexpected(fwd_r.error());

        auto s_r = map_result(ccop::greedy_sample(output.logits, batch.logits_indices,
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
            wr.item_index = i;
            wr.seq_id = batch.item_seq_ids[i];
            wr.kind = batch.item_kinds[i];
            const bool sampled = batch.sample_flags[i];
            wr.sampled_tokens =
                sampled ? std::vector<int32_t>{tokens_host[i]} : std::vector<int32_t>{};
            wr.tokens_consumed = batch.item_token_counts[i];
            wr.eos =
                sampled && sampling.eos_token_id >= 0 && tokens_host[i] == sampling.eos_token_id;
            results.push_back(std::move(wr));
        }

        return results;
    }
};

}  // namespace ccinfer
