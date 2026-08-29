#include "model/qwen3/qwen3_model.h"

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "backend/backend.h"
#include "cache/block_storage.h"

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
    return model;
}

Qwen3Model::Qwen3Model(ModelConfig config, Qwen3Weights weights, RopeCache rope_cache)
    : config_(std::move(config)),
      weights_(std::move(weights)),
      rope_cache_(std::move(rope_cache)) {}

Result<void> Qwen3Model::forward(const ForwardInput& input, ForwardOutput& output,
                                 Backend& backend) {
    assert(output.logits.valid());
    assert(input.num_tokens_ > 0);
    // Exactly one of token_ids / input_embeds must be valid.
    assert(input.token_ids.valid() != input.input_embeds.valid());
    assert(input.block_storage_ != nullptr);
    assert(input.max_position_id_ >= 0 && input.max_position_id_ < rope_cache_.max_position());

    const int T = input.num_tokens_;
    const int D = config_.d_model_;
    const int nq = config_.n_q_heads_;
    const int nkv = config_.n_kv_heads_;
    const int hd = config_.head_dim_;
    const int d_ff = config_.d_ff_;
    const int n_layers = config_.n_layers_;
    const float eps = config_.rms_norm_eps_;

    const int qkv_dim = (nq + 2 * nkv) * hd;
    const int attn_dim = nq * hd;

    const std::int64_t T64 = T;

    auto alloc_tensor = [&](std::initializer_list<std::int64_t> shape) -> Result<Tensor> {
        return Tensor::empty(backend, ccop::DType::kBFloat16, shape);
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

    const ccop::ExecutionContext ctx = backend.context();

    if (input.token_ids.valid()) {
        auto r = map_result(ccop::embed(weights_.embed, input.token_ids, &hidden, ctx));
        if (!r) return std::unexpected(r.error());
    } else {
        auto r = backend.memcpy_d2d(hidden.data(), input.input_embeds.data(), hidden.nbytes());
        if (!r) return std::unexpected(r.error());
    }

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));

    for (int l = 0; l < n_layers; ++l) {
        const auto& lw = weights_.layers_[static_cast<std::size_t>(l)];

        {
            auto r = map_result(ccop::rms_norm(&normed, hidden, lw.rms_attn, eps, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = map_result(ccop::gemm(&qkv_out, normed, lw.qkv, false, true, 1.0f, 0.0f, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = map_result(ccop::split_qkv(qkv_out, &q, &k, &v, ctx));
            if (!r) return std::unexpected(r.error());
        }

        if (lw.q_norm.valid()) {
            auto r = map_result(ccop::rms_norm(&q_norm_2d, q_norm_2d, lw.q_norm, eps, ctx));
            if (!r) return std::unexpected(r.error());
        }
        if (lw.k_norm.valid()) {
            auto r = map_result(ccop::rms_norm(&k_norm_2d, k_norm_2d, lw.k_norm, eps, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = map_result(ccop::rope(&q, &k, input.positions, rope_cache_.tensor(), hd, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            // The whole batch enters write_kv_cache once.  DecodeOneToken with
            // write_kv=false uses -1 slots, so the operator skips them.
            auto k_cache = input.block_storage_->k_layer_tensor(l);
            auto v_cache = input.block_storage_->v_layer_tensor(l);
            auto r =
                map_result(ccop::write_kv_cache(k, v, &k_cache, &v_cache, input.slot_mapping, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto k_blocks = input.block_storage_->k_block_tensor(l);
            auto v_blocks = input.block_storage_->v_block_tensor(l);
            if (input.mode_ == ForwardMode::Decode) {
                auto r = map_result(ccop::decode_attention(q, k_blocks, v_blocks, input.block_table,
                                                           input.context_lens, &attn_out_3d,
                                                           attn_scale, ctx));
                if (!r) return std::unexpected(r.error());
            } else {
                auto r = map_result(ccop::prefill_attention(
                    q, k_blocks, v_blocks, input.block_table, input.query_start_loc,
                    input.context_lens, &attn_out_3d, attn_scale, ctx));
                if (!r) return std::unexpected(r.error());
            }
        }

        {
            auto r =
                map_result(ccop::gemm(&next_hidden, attn_out, lw.o, false, true, 1.0f, 0.0f, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = map_result(ccop::element_add(&next_hidden_flat, hidden_flat, ctx));
            if (!r) return std::unexpected(r.error());
        }
        std::swap(hidden, next_hidden);
        std::swap(hidden_flat, next_hidden_flat);

        {
            auto r = map_result(ccop::rms_norm(&normed, hidden, lw.rms_ffn, eps, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = map_result(ccop::gemm(&gate, normed, lw.gate, false, true, 1.0f, 0.0f, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = map_result(ccop::gemm(&up, normed, lw.up, false, true, 1.0f, 0.0f, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = map_result(ccop::silu_mul(&ffn_act_flat, gate_flat, up_flat, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = map_result(
                ccop::gemm(&next_hidden, ffn_act, lw.down, false, true, 1.0f, 0.0f, ctx));
            if (!r) return std::unexpected(r.error());
        }

        {
            auto r = map_result(ccop::element_add(&next_hidden_flat, hidden_flat, ctx));
            if (!r) return std::unexpected(r.error());
        }
        std::swap(hidden, next_hidden);
        std::swap(hidden_flat, next_hidden_flat);
    }

    {
        auto r = map_result(ccop::rms_norm(&normed, hidden, weights_.rms_final, eps, ctx));
        if (!r) return std::unexpected(r.error());
    }

    if (input.num_logits_ > 0) {
        assert(input.logits_indices_host.size() == static_cast<std::size_t>(input.num_logits_));
        for (int i = 0; i < input.num_logits_; ++i) {
            const int row = input.logits_indices_host[static_cast<std::size_t>(i)];
            assert(row >= 0 && row < T);
            auto normed_row = normed.slice(0, row, row + 1);
            auto logits_row = output.logits.slice(0, i, i + 1);
            auto r = map_result(ccop::gemm(&logits_row, normed_row, weights_.lm_head, false, true,
                                           1.0f, 0.0f, ctx));
            if (!r) return std::unexpected(r.error());
        }
    }

    return {};
}

}  // namespace ccinfer
