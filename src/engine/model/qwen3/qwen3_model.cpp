#include "engine/model/qwen3/qwen3_model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>

#include "engine/backend/backend.h"
#include "engine/backend/device_buffer.h"
#include "engine/cache/kv_cache_storage.h"

namespace ccinfer {
namespace engine {

Result<std::unique_ptr<Model>> Qwen3Model::create(const ModelConfig& config,
                                                  const WeightLoader& loader,
                                                  DefaultBackend& backend) {
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
                                 DefaultBackend& backend) {
    // --- Validate inputs ---
    if (output.logits_ == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
    if (input.num_tokens_ <= 0) return std::unexpected(ErrorCode::InvalidArgument);
    if (input.positions_ == nullptr) return std::unexpected(ErrorCode::InvalidArgument);

    // token_ids_ and input_embeds_ must be mutually exclusive.
    if ((input.token_ids_ == nullptr) == (input.input_embeds_ == nullptr)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    if (input.kv_storage_ == nullptr || input.slot_mapping_ == nullptr ||
        input.block_table_ == nullptr || input.context_lens_ == nullptr || input.batch_size_ <= 0 ||
        input.max_blocks_per_req_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    const int cache_block_size = input.kv_storage_->block_size();
    if (cache_block_size <= 0) return std::unexpected(ErrorCode::InvalidArgument);

    if (input.mode_ == ForwardMode::Prefill && input.query_start_loc_ == nullptr) {
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
    if (D != nq * hd) return std::unexpected(ErrorCode::ModelShapeMismatch);
    if (nq % nkv != 0) return std::unexpected(ErrorCode::ModelShapeMismatch);
    if (input.mode_ == ForwardMode::Decode) {
        if (T != B) return std::unexpected(ErrorCode::ModelShapeMismatch);
    }
    if (static_cast<int>(weights_.layers_.size()) < n_layers) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    const int qkv_dim = (nq + 2 * nkv) * hd;
    const int attn_dim = nq * hd;
    const int max_slots = input.kv_storage_->max_slots();
    const int64_t T64 = T;
    const int64_t D64 = D;

    // --- Allocate workspace buffers ---
    auto alloc = [&](std::size_t bytes) -> Result<std::unique_ptr<DeviceBuffer>> {
        auto r = backend.allocate_buffer(bytes);
        if (!r) return std::unexpected(r.error());
        return std::move(*r);
    };

    auto hidden_a = alloc(static_cast<std::size_t>(T) * D * sizeof(__nv_bfloat16));
    if (!hidden_a) return std::unexpected(hidden_a.error());
    auto hidden_b = alloc(static_cast<std::size_t>(T) * D * sizeof(__nv_bfloat16));
    if (!hidden_b) return std::unexpected(hidden_b.error());
    auto normed = alloc(static_cast<std::size_t>(T) * D * sizeof(__nv_bfloat16));
    if (!normed) return std::unexpected(normed.error());
    auto qkv_out = alloc(static_cast<std::size_t>(T) * qkv_dim * sizeof(__nv_bfloat16));
    if (!qkv_out) return std::unexpected(qkv_out.error());
    auto q_buf = alloc(static_cast<std::size_t>(T) * nq * hd * sizeof(__nv_bfloat16));
    if (!q_buf) return std::unexpected(q_buf.error());
    auto k_buf = alloc(static_cast<std::size_t>(T) * nkv * hd * sizeof(__nv_bfloat16));
    if (!k_buf) return std::unexpected(k_buf.error());
    auto v_buf = alloc(static_cast<std::size_t>(T) * nkv * hd * sizeof(__nv_bfloat16));
    if (!v_buf) return std::unexpected(v_buf.error());
    auto attn_out = alloc(static_cast<std::size_t>(T) * attn_dim * sizeof(__nv_bfloat16));
    if (!attn_out) return std::unexpected(attn_out.error());
    auto gate = alloc(static_cast<std::size_t>(T) * d_ff * sizeof(__nv_bfloat16));
    if (!gate) return std::unexpected(gate.error());
    auto up_buf = alloc(static_cast<std::size_t>(T) * d_ff * sizeof(__nv_bfloat16));
    if (!up_buf) return std::unexpected(up_buf.error());
    auto ffn_act = alloc(static_cast<std::size_t>(T) * d_ff * sizeof(__nv_bfloat16));
    if (!ffn_act) return std::unexpected(ffn_act.error());

    // --- Embed ---
    if (input.token_ids_ != nullptr) {
        auto er = backend.embed(EmbedParams{
            .embed_table_ = weights_.embed_->data(),
            .token_ids_ = input.token_ids_,
            .input_embeds_ = (*hidden_a)->data(),
            .num_tokens_ = T,
            .d_model_ = D,
        });
        if (!er) return er;
    } else {
        auto r = backend.memcpy_d2d((*hidden_a)->data(), input.input_embeds_,
                                    static_cast<std::size_t>(T) * D * sizeof(__nv_bfloat16));
        if (!r) return std::unexpected(r.error());
    }

    auto* hidden = static_cast<__nv_bfloat16*>((*hidden_a)->data());
    auto* next_hidden = static_cast<__nv_bfloat16*>((*hidden_b)->data());

    // --- Transformer layers ---
    for (int l = 0; l < n_layers; ++l) {
        const auto& lw = weights_.layers_[static_cast<std::size_t>(l)];

        // RMSNorm (attention)
        {
            auto r = backend.template rms_norm<__nv_bfloat16>(RmsNormParams{
                .input_ = hidden,
                .weight_ = lw.rms_attn_->data(),
                .output_ = (*normed)->data(),
                .rows_ = T,
                .dim_ = D,
                .eps_ = eps,
            });
            if (!r) return r;
        }

        // QKV projection: qkv_out[T,qkv_dim] = normed[T,D] @ qkv_weight[qkv_dim,D]^T
        {
            auto r = backend.template gemm<__nv_bfloat16>(GemmParams{
                .a_ = (*normed)->data(),
                .b_ = lw.qkv_->data(),
                .c_ = (*qkv_out)->data(),
                .m_ = T,
                .n_ = qkv_dim,
                .k_ = D,
                .lda_ = D,
                .ldb_ = D,
                .ldc_ = qkv_dim,
                .trans_b_ = true,
            });
            if (!r) return r;
        }

        // Split QKV
        {
            auto r = backend.template split_qkv<__nv_bfloat16>(SplitQkvParams{
                .qkv_ = (*qkv_out)->data(),
                .q_ = (*q_buf)->data(),
                .k_ = (*k_buf)->data(),
                .v_ = (*v_buf)->data(),
                .num_tokens_ = T,
                .num_q_heads_ = nq,
                .num_kv_heads_ = nkv,
                .head_dim_ = hd,
            });
            if (!r) return r;
        }

        // Q/K norm (Qwen3) — verified in-place safe.
        if (lw.q_norm_) {
            auto r = backend.template rms_norm<__nv_bfloat16>(RmsNormParams{
                .input_ = (*q_buf)->data(),
                .weight_ = lw.q_norm_->data(),
                .output_ = (*q_buf)->data(),
                .rows_ = T * nq,
                .dim_ = hd,
                .eps_ = eps,
            });
            if (!r) return r;
        }
        if (lw.k_norm_) {
            auto r = backend.template rms_norm<__nv_bfloat16>(RmsNormParams{
                .input_ = (*k_buf)->data(),
                .weight_ = lw.k_norm_->data(),
                .output_ = (*k_buf)->data(),
                .rows_ = T * nkv,
                .dim_ = hd,
                .eps_ = eps,
            });
            if (!r) return r;
        }

        // RoPE
        {
            auto r = backend.template rope<__nv_bfloat16>(RopeParams{
                .q_ = (*q_buf)->data(),
                .k_ = (*k_buf)->data(),
                .positions_ = input.positions_,
                .rope_cache_ = rope_cache_.data(),
                .num_tokens_ = T,
                .num_q_heads_ = nq,
                .num_kv_heads_ = nkv,
                .head_dim_ = hd,
                .rotary_dim_ = hd,
                .rope_cache_max_position_ = rope_cache_.max_position(),
            });
            if (!r) return r;
        }

        // Write KV cache
        {
            auto r = backend.template write_kv_cache<__nv_bfloat16>(WriteKVCacheParams{
                .k_new_ = (*k_buf)->data(),
                .v_new_ = (*v_buf)->data(),
                .k_cache_ = input.kv_storage_->k_layer(l),
                .v_cache_ = input.kv_storage_->v_layer(l),
                .slot_mapping_ = input.slot_mapping_,
                .total_tokens_ = T,
                .num_kv_heads_ = nkv,
                .head_dim_ = hd,
                .max_slots_ = max_slots,
            });
            if (!r) return r;
        }

        // Paged attention
        if (input.mode_ == ForwardMode::Decode) {
            auto r = backend.template decode_attention<__nv_bfloat16>(DecodeAttnParams{
                .q_ = (*q_buf)->data(),
                .k_cache_ = input.kv_storage_->k_layer(l),
                .v_cache_ = input.kv_storage_->v_layer(l),
                .block_table_ = input.block_table_,
                .context_lens_ = input.context_lens_,
                .output_ = (*attn_out)->data(),
                .batch_size_ = B,
                .max_blocks_per_req_ = input.max_blocks_per_req_,
                .num_q_heads_ = nq,
                .num_kv_heads_ = nkv,
                .head_dim_ = hd,
                .cache_block_size_ = cache_block_size,
            });
            if (!r) return r;
        } else {
            auto r = backend.template prefill_attention<__nv_bfloat16>(PrefillAttnParams{
                .q_ = (*q_buf)->data(),
                .k_cache_ = input.kv_storage_->k_layer(l),
                .v_cache_ = input.kv_storage_->v_layer(l),
                .block_table_ = input.block_table_,
                .query_start_loc_ = input.query_start_loc_,
                .context_lens_ = input.context_lens_,
                .output_ = (*attn_out)->data(),
                .num_tokens_ = T,
                .batch_size_ = B,
                .max_blocks_per_req_ = input.max_blocks_per_req_,
                .num_q_heads_ = nq,
                .num_kv_heads_ = nkv,
                .head_dim_ = hd,
                .cache_block_size_ = cache_block_size,
            });
            if (!r) return r;
        }

        // Output projection: next_hidden[T,D] = attn_out[T,attn_dim] @ o_weight[D,attn_dim]^T
        {
            auto r = backend.template gemm<__nv_bfloat16>(GemmParams{
                .a_ = (*attn_out)->data(),
                .b_ = lw.o_->data(),
                .c_ = next_hidden,
                .m_ = T,
                .n_ = D,
                .k_ = attn_dim,
                .lda_ = attn_dim,
                .ldb_ = attn_dim,
                .ldc_ = D,
                .trans_b_ = true,
            });
            if (!r) return r;
        }

        // Residual: next_hidden += hidden
        {
            auto r = backend.template element_add<__nv_bfloat16>(ElementAddParams{
                .dst_ = next_hidden,
                .src_ = hidden,
                .n_ = T64 * D64,
            });
            if (!r) return r;
        }
        std::swap(hidden, next_hidden);

        // FFN: RMSNorm
        {
            auto r = backend.template rms_norm<__nv_bfloat16>(RmsNormParams{
                .input_ = hidden,
                .weight_ = lw.rms_ffn_->data(),
                .output_ = (*normed)->data(),
                .rows_ = T,
                .dim_ = D,
                .eps_ = eps,
            });
            if (!r) return r;
        }

        // Gate projection: gate[T,d_ff] = normed[T,D] @ gate_weight[d_ff,D]^T
        {
            auto r = backend.template gemm<__nv_bfloat16>(GemmParams{
                .a_ = (*normed)->data(),
                .b_ = lw.gate_->data(),
                .c_ = (*gate)->data(),
                .m_ = T,
                .n_ = d_ff,
                .k_ = D,
                .lda_ = D,
                .ldb_ = D,
                .ldc_ = d_ff,
                .trans_b_ = true,
            });
            if (!r) return r;
        }

        // Up projection: up[T,d_ff] = normed[T,D] @ up_weight[d_ff,D]^T
        {
            auto r = backend.template gemm<__nv_bfloat16>(GemmParams{
                .a_ = (*normed)->data(),
                .b_ = lw.up_->data(),
                .c_ = (*up_buf)->data(),
                .m_ = T,
                .n_ = d_ff,
                .k_ = D,
                .lda_ = D,
                .ldb_ = D,
                .ldc_ = d_ff,
                .trans_b_ = true,
            });
            if (!r) return r;
        }

        // ffn_act = silu(gate) * up
        {
            auto r = backend.template silu_mul<__nv_bfloat16>(SiluMulParams{
                .gate_ = (*gate)->data(),
                .up_ = (*up_buf)->data(),
                .output_ = (*ffn_act)->data(),
                .n_ = T64 * d_ff,
            });
            if (!r) return r;
        }

        // Down projection: next_hidden[T,D] = ffn_act[T,d_ff] @ down_weight[D,d_ff]^T
        {
            auto r = backend.template gemm<__nv_bfloat16>(GemmParams{
                .a_ = (*ffn_act)->data(),
                .b_ = lw.down_->data(),
                .c_ = next_hidden,
                .m_ = T,
                .n_ = D,
                .k_ = d_ff,
                .lda_ = d_ff,
                .ldb_ = d_ff,
                .ldc_ = D,
                .trans_b_ = true,
            });
            if (!r) return r;
        }

        // Residual: next_hidden += hidden
        {
            auto r = backend.template element_add<__nv_bfloat16>(ElementAddParams{
                .dst_ = next_hidden,
                .src_ = hidden,
                .n_ = T64 * D64,
            });
            if (!r) return r;
        }
        std::swap(hidden, next_hidden);
    }

    // --- Final RMSNorm ---
    {
        auto r = backend.template rms_norm<__nv_bfloat16>(RmsNormParams{
            .input_ = hidden,
            .weight_ = weights_.rms_final_->data(),
            .output_ = (*normed)->data(),
            .rows_ = T,
            .dim_ = D,
            .eps_ = eps,
        });
        if (!r) return r;
    }

    // --- LM head: logits[T,V] = normed[T,D] @ lm_head_weight[V,D]^T ---
    {
        auto r = backend.gemm_logits(GemmParams{
            .a_ = (*normed)->data(),
            .b_ = weights_.lm_head_->data(),
            .c_ = output.logits_,
            .m_ = T,
            .n_ = V,
            .k_ = D,
            .lda_ = D,
            .ldb_ = D,
            .ldc_ = V,
            .trans_b_ = true,
        });
        if (!r) return r;
    }

    return {};
}

}  // namespace engine
}  // namespace ccinfer
