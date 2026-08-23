#include "model/qwen3/qwen3_model.h"

#include <cmath>
#include <cstdint>
#include <utility>

#include <cuda_runtime.h>

#include "backend/backend.h"
#include "cache/kv_cache_manager.h"

namespace ccinfer {

Result<std::unique_ptr<Model>> Qwen3Model::create(const ModelConfig& config,
                                                  const WeightLoader& loader, Backend& backend) {
    auto weights = Qwen3Weights::load(backend, config, loader);
    if (!weights) return std::unexpected(weights.error());

    auto rope_cache =
        RopeCache::create(config.max_seq_len_, config.head_dim_, config.rope_theta_, backend);
    if (!rope_cache) return std::unexpected(rope_cache.error());

    std::unique_ptr<Model> model =
        std::make_unique<Qwen3Model>(config, std::move(*weights), std::move(*rope_cache));
    return std::move(model);
}

Qwen3Model::Qwen3Model(ModelConfig config, Qwen3Weights weights, RopeCache rope_cache)
    : config_(std::move(config)),
      weights_(std::move(weights)),
      rope_cache_(std::move(rope_cache)) {}

Result<void> Qwen3Model::forward(const ForwardInput& input, ForwardOutput& output,
                                 Backend& backend) {
    if (!output.logits.valid()) return std::unexpected(ErrorCode::InvalidArgument);
    if (input.num_tokens_ <= 0) return std::unexpected(ErrorCode::InvalidArgument);

    // Exactly one of token_ids / input_embeds must be valid.
    if (input.token_ids.valid() == input.input_embeds.valid()) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    if (input.kv_mgr_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    if (input.max_position_id_ < 0 || input.max_position_id_ >= rope_cache_.max_position()) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    const int T = input.num_tokens_;
    const int B = input.batch_size_;
    const int D = config_.d_model_;
    const int nq = config_.n_q_heads_;
    const int nkv = config_.n_kv_heads_;
    const int hd = config_.head_dim_;
    const int d_ff = config_.d_ff_;
    const int V = config_.vocab_size_;
    const int n_layers = config_.n_layers_;
    const float eps = config_.rms_norm_eps_;

    if (D <= 0 || nq <= 0 || nkv <= 0 || hd <= 0 || d_ff <= 0 || V <= 0 || n_layers <= 0) {
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    }
    if (nq % nkv != 0) return std::unexpected(ErrorCode::ModelShapeMismatch);
    if (input.mode_ == ForwardMode::Decode) {
        if (T != B) return std::unexpected(ErrorCode::ModelShapeMismatch);
    }
    if (static_cast<int>(weights_.layers_.size()) < n_layers) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    const int qkv_dim = (nq + 2 * nkv) * hd;
    const int attn_dim = nq * hd;

    const std::int64_t T64 = T;
    const std::int64_t d_ff64 = d_ff;

    auto alloc_tensor = [&](std::initializer_list<std::int64_t> shape) -> Result<Tensor> {
        return Tensor::empty(backend, ops::DType::kBFloat16, shape);
    };

    auto hidden_r = alloc_tensor({T, D});
    if (!hidden_r) return std::unexpected(hidden_r.error());
    Tensor hidden = std::move(*hidden_r);
    Tensor hidden_flat = hidden.flat();

    auto next_hidden_r = alloc_tensor({T, D});
    if (!next_hidden_r) return std::unexpected(next_hidden_r.error());
    Tensor next_hidden = std::move(*next_hidden_r);
    Tensor next_hidden_flat = next_hidden.flat();

    auto normed_r = alloc_tensor({T, D});
    if (!normed_r) return std::unexpected(normed_r.error());
    Tensor normed = std::move(*normed_r);

    auto qkv_out_r = alloc_tensor({T, qkv_dim});
    if (!qkv_out_r) return std::unexpected(qkv_out_r.error());
    Tensor qkv_out = std::move(*qkv_out_r);

    auto q_r = alloc_tensor({T, nq, hd});
    if (!q_r) return std::unexpected(q_r.error());
    Tensor q = std::move(*q_r);
    Tensor q_norm_2d = q.view({T64 * nq, hd});

    auto k_r = alloc_tensor({T, nkv, hd});
    if (!k_r) return std::unexpected(k_r.error());
    Tensor k = std::move(*k_r);
    Tensor k_norm_2d = k.view({T64 * nkv, hd});

    auto v_r = alloc_tensor({T, nkv, hd});
    if (!v_r) return std::unexpected(v_r.error());
    Tensor v = std::move(*v_r);

    auto attn_out_r = alloc_tensor({T, attn_dim});
    if (!attn_out_r) return std::unexpected(attn_out_r.error());
    Tensor attn_out = std::move(*attn_out_r);
    Tensor attn_out_3d = attn_out.view({T, nq, hd});

    auto gate_r = alloc_tensor({T, d_ff});
    if (!gate_r) return std::unexpected(gate_r.error());
    Tensor gate = std::move(*gate_r);
    Tensor gate_flat = gate.flat();

    auto up_r = alloc_tensor({T, d_ff});
    if (!up_r) return std::unexpected(up_r.error());
    Tensor up = std::move(*up_r);
    Tensor up_flat = up.flat();

    auto ffn_act_r = alloc_tensor({T, d_ff});
    if (!ffn_act_r) return std::unexpected(ffn_act_r.error());
    Tensor ffn_act = std::move(*ffn_act_r);
    Tensor ffn_act_flat = ffn_act.flat();

    const ops::ExecutionContext ctx = backend.context();

    if (input.token_ids.valid()) {
        auto r = ops::map_result(ops::embed(weights_.embed, input.token_ids, &hidden, ctx));
        if (!r) return std::unexpected(r.error());
    } else {
        auto r = backend.memcpy_d2d(hidden.data(), input.input_embeds.data(), hidden.nbytes());
        if (!r) return std::unexpected(r.error());
    }

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));

    for (int l = 0; l < n_layers; ++l) {
        const auto& lw = weights_.layers_[static_cast<std::size_t>(l)];

        {
            auto r = ops::map_result(ops::rms_norm(&normed, hidden, lw.rms_attn, eps, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r =
                ops::map_result(ops::gemm(&qkv_out, normed, lw.qkv, false, true, 1.0f, 0.0f, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = ops::map_result(ops::split_qkv(qkv_out, &q, &k, &v, ctx));
            if (!r) return std::unexpected(r.error());
        }

        if (lw.q_norm.valid()) {
            auto r = ops::map_result(ops::rms_norm(&q_norm_2d, q_norm_2d, lw.q_norm, eps, ctx));
            if (!r) return std::unexpected(r.error());
        }
        if (lw.k_norm.valid()) {
            auto r = ops::map_result(ops::rms_norm(&k_norm_2d, k_norm_2d, lw.k_norm, eps, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r =
                ops::map_result(ops::rope(&q, &k, input.positions, rope_cache_.tensor(), hd, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            // The whole batch enters write_kv_cache once.  DecodeOneToken with
            // write_kv=false uses -1 slots, so the operator skips them.
            auto k_cache = input.kv_mgr_->k_cache(l);
            auto v_cache = input.kv_mgr_->v_cache(l);
            auto r = ops::map_result(
                ops::write_kv_cache(k, v, &k_cache, &v_cache, input.slot_mapping, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto k_blocks = input.kv_mgr_->k_cache_blocks(l);
            auto v_blocks = input.kv_mgr_->v_cache_blocks(l);
            if (input.mode_ == ForwardMode::Decode) {
                auto r = ops::map_result(
                    ops::decode_attention(q, k_blocks, v_blocks, input.block_table,
                                          input.context_lens, &attn_out_3d, attn_scale, ctx));
                if (!r) return std::unexpected(r.error());
            } else {
                auto r = ops::map_result(ops::prefill_attention(
                    q, k_blocks, v_blocks, input.block_table, input.query_start_loc,
                    input.context_lens, &attn_out_3d, attn_scale, ctx));
                if (!r) return std::unexpected(r.error());
            }
        }

        {
            auto r = ops::map_result(
                ops::gemm(&next_hidden, attn_out, lw.o, false, true, 1.0f, 0.0f, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = ops::map_result(ops::element_add(&next_hidden_flat, hidden_flat, ctx));
            if (!r) return std::unexpected(r.error());
        }
        std::swap(hidden, next_hidden);
        std::swap(hidden_flat, next_hidden_flat);

        {
            auto r = ops::map_result(ops::rms_norm(&normed, hidden, lw.rms_ffn, eps, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r =
                ops::map_result(ops::gemm(&gate, normed, lw.gate, false, true, 1.0f, 0.0f, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = ops::map_result(ops::gemm(&up, normed, lw.up, false, true, 1.0f, 0.0f, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = ops::map_result(ops::silu_mul(&ffn_act_flat, gate_flat, up_flat, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = ops::map_result(
                ops::gemm(&next_hidden, ffn_act, lw.down, false, true, 1.0f, 0.0f, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = ops::map_result(ops::element_add(&next_hidden_flat, hidden_flat, ctx));
            if (!r) return std::unexpected(r.error());
        }
        std::swap(hidden, next_hidden);
        std::swap(hidden_flat, next_hidden_flat);
    }

    {
        auto r = ops::map_result(ops::rms_norm(&normed, hidden, weights_.rms_final, eps, ctx));
        if (!r) return std::unexpected(r.error());
    }

    {
        auto r = ops::map_result(
            ops::gemm(&output.logits, normed, weights_.lm_head, false, true, 1.0f, 0.0f, ctx));
        if (!r) return std::unexpected(r.error());
    }

    return {};
}

}  // namespace ccinfer
