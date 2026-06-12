#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "base/result.h"
#include "backend/params.h"
#include "cache/kv_cache_manager.h"
#include "backend/default_backend.h"
#include "core/traits.h"
#include "base/execution.h"
#include "model/config.h"
#include "model/model.h"

namespace ccinfer {

class ModelRunner {
public:
    template <typename Traits>
    static Result<std::vector<WorkItemResult>> inference(Model& model, const PhysicalBatch& batch,
                                                         DefaultBackend& backend,
                                                         KVCacheManager& kv_mgr,
                                                         const SamplingParams& sampling = {}) {
        static_assert(runner_traits_valid_v<Traits>, "RunnerTraits has unknown dtype tags");

        // Phase 4.1: only BF16 weights / activations / KV + FP32 logits.
        if constexpr (!std::is_same_v<typename Traits::WeightTag, BFloat16Tag> ||
                      !std::is_same_v<typename Traits::KVTag, BFloat16Tag> ||
                      !std::is_same_v<typename Traits::ActivationTag, BFloat16Tag> ||
                      !std::is_same_v<typename Traits::LogitsTag, Float32Tag>) {
            (void)Traits{};
            return std::unexpected(ErrorCode::Unsupported);
        }

        // --- Validate physical batch ---
        if (!batch.token_ids || !batch.positions || !batch.slot_mapping || !batch.block_table ||
            !batch.query_start_loc || !batch.context_lens || !batch.logits_indices) {
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

        // --- Validate sampling params ---
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

        // --- D2H: query_start_loc (validate + reuse later) ---
        std::vector<int32_t> qsl_host(B + 1);
        {
            auto r = backend.memcpy_d2h(qsl_host.data(), batch.query_start_loc->data(),
                                        static_cast<std::size_t>(B + 1) * sizeof(int32_t));
            if (!r) return std::unexpected(r.error());
            if (qsl_host[0] != 0) return std::unexpected(ErrorCode::InvalidArgument);
            if (qsl_host[B] != T) return std::unexpected(ErrorCode::InvalidArgument);
            for (int i = 0; i < B; ++i) {
                int len = qsl_host[i + 1] - qsl_host[i];
                if (len <= 0) return std::unexpected(ErrorCode::InvalidArgument);
                if (batch.mode == ForwardMode::Decode && len != 1)
                    return std::unexpected(ErrorCode::InvalidArgument);
                if (batch.item_kinds[i] == WorkKind::DecodeOneToken && len != 1)
                    return std::unexpected(ErrorCode::InvalidArgument);
            }
        }

        // --- D2H: positions (validate + compute max_position_id) ---
        int max_pos = 0;
        {
            std::vector<int32_t> pos_host(T);
            auto r = backend.memcpy_d2h(pos_host.data(), batch.positions->data(),
                                        static_cast<std::size_t>(T) * sizeof(int32_t));
            if (!r) return std::unexpected(r.error());
            for (int t = 0; t < T; ++t) {
                if (pos_host[t] < 0) return std::unexpected(ErrorCode::InvalidArgument);
                if (pos_host[t] > max_pos) max_pos = pos_host[t];
            }
        }

        // --- D2H: context_lens (validate) ---
        {
            const int block_sz = kv_mgr.block_size();
            const int max_slots = kv_mgr.max_slots();
            if (block_sz <= 0 || max_slots <= 0) return std::unexpected(ErrorCode::InvalidArgument);
            if (batch.max_blocks_per_req > max_slots / block_sz)
                return std::unexpected(ErrorCode::InvalidArgument);
            const int max_ctx = batch.max_blocks_per_req * block_sz;

            std::vector<int32_t> ctx_host(B);
            auto r = backend.memcpy_d2h(ctx_host.data(), batch.context_lens->data(),
                                        static_cast<std::size_t>(B) * sizeof(int32_t));
            if (!r) return std::unexpected(r.error());
            for (int i = 0; i < B; ++i) {
                int ctx = ctx_host[i];
                int qlen = qsl_host[i + 1] - qsl_host[i];
                if (ctx <= 0) return std::unexpected(ErrorCode::InvalidArgument);
                if (ctx < qlen) return std::unexpected(ErrorCode::InvalidArgument);
                if (ctx > max_ctx) return std::unexpected(ErrorCode::InvalidArgument);
            }
        }

        // --- D2H: logits_indices (validate, before forward to avoid KV side-effects) ---
        std::vector<int32_t> li_host(B);
        {
            auto r = backend.memcpy_d2h(li_host.data(), batch.logits_indices->data(),
                                        static_cast<std::size_t>(B) * sizeof(int32_t));
            if (!r) return std::unexpected(r.error());
            for (int i = 0; i < B; ++i) {
                if (li_host[i] == -1) continue;  // skip-sample sentinel
                if (li_host[i] < 0 || li_host[i] >= T)
                    return std::unexpected(ErrorCode::InvalidArgument);
                if (batch.mode == ForwardMode::Decode && li_host[i] != i)
                    return std::unexpected(ErrorCode::InvalidArgument);
                if ((batch.mode == ForwardMode::Prefill || batch.mode == ForwardMode::Mixed) &&
                    li_host[i] != qsl_host[i + 1] - 1)
                    return std::unexpected(ErrorCode::InvalidArgument);
            }
        }

        // --- Build ForwardInput ---
        ForwardInput input;
        input.token_ids_ = static_cast<const int32_t*>(batch.token_ids->data());
        input.positions_ = static_cast<const int32_t*>(batch.positions->data());
        input.num_tokens_ = T;
        input.max_position_id_ = max_pos;
        input.kv_mgr_ = &kv_mgr;
        input.slot_mapping_ = static_cast<const int32_t*>(batch.slot_mapping->data());
        input.block_table_ = static_cast<const int32_t*>(batch.block_table->data());
        input.query_start_loc_ = static_cast<const int32_t*>(batch.query_start_loc->data());
        input.context_lens_ = static_cast<const int32_t*>(batch.context_lens->data());
        input.batch_size_ = B;
        input.max_blocks_per_req_ = batch.max_blocks_per_req;
        input.mode_ = batch.mode;

        constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
        const std::size_t T_size = static_cast<std::size_t>(T);
        const std::size_t V_size = static_cast<std::size_t>(V);
        if (V_size > kMax / T_size) return std::unexpected(ErrorCode::InvalidArgument);
        const std::size_t logits_elems = T_size * V_size;
        if (logits_elems > kMax / sizeof(float)) return std::unexpected(ErrorCode::InvalidArgument);
        auto logits_r = backend.allocate_buffer(logits_elems * sizeof(float));
        if (!logits_r) return std::unexpected(logits_r.error());
        auto logits = std::move(*logits_r);

        auto tokens_r = backend.allocate_buffer(static_cast<std::size_t>(B) * sizeof(int32_t));
        if (!tokens_r) return std::unexpected(tokens_r.error());
        auto tokens_gpu = std::move(*tokens_r);

        ForwardOutput output;
        output.logits_ = logits->data();
        output.tokens_out_ = static_cast<int32_t*>(tokens_gpu->data());
        auto fwd_r = model.forward(input, output, backend);
        if (!fwd_r) return std::unexpected(fwd_r.error());

        SampleParams sp;
        sp.logits_ = static_cast<const float*>(logits->data());
        sp.logits_indices_ = static_cast<const int32_t*>(batch.logits_indices->data());
        sp.tokens_out_ = static_cast<int32_t*>(tokens_gpu->data());
        sp.num_tokens_ = T;
        sp.batch_size_ = B;
        sp.vocab_size_ = V;
        sp.top_k_ = 0;
        sp.top_p_ = 1.0f;
        sp.temperature_ = 0.0f;
        sp.seed_ = sampling.seed;

        auto s_r = backend.sample(sp);
        if (!s_r) return std::unexpected(s_r.error());

        // D2H: sampled tokens.
        std::vector<int32_t> tokens_host(B);
        {
            auto r = backend.memcpy_d2h(tokens_host.data(), tokens_gpu->data(),
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
            // li_host[i] == -1 means skip-sample — intermediate prefill chunk.
            wr.sampled_tokens =
                (li_host[i] >= 0) ? std::vector<int32_t>{tokens_host[i]} : std::vector<int32_t>{};
            wr.tokens_consumed = qsl_host[i + 1] - qsl_host[i];
            wr.eos = false;
            results.push_back(std::move(wr));
        }

        return results;
    }
};

}  // namespace ccinfer
