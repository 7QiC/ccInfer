#include "engine/model/qwen3/qwen3_model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "engine/backend/backend.h"
#include "engine/backend/cuda/cuda_utils.h"
#include "engine/backend/device_buffer.h"
#include "engine/kernel/cuda_kernels.h"

namespace ccinfer {
namespace engine {

Result<std::unique_ptr<Model>> Qwen3Model::create(const ModelConfig& config,
                                                  const WeightLoader& loader,
                                                  DeviceBackend& backend) {
    auto weights = Qwen3Weights::load(backend, config, loader);
    if (!weights) return std::unexpected(weights.error());

    auto rope_cache = RopeCache::create(config.max_seq_len_, config.head_dim_, config.rope_theta_, backend);
    if (!rope_cache) return std::unexpected(rope_cache.error());

    return std::make_unique<Qwen3Model>(config, std::move(*weights), std::move(*rope_cache));
}

Qwen3Model::Qwen3Model(ModelConfig config, Qwen3Weights weights, RopeCache rope_cache)
    : config_(std::move(config)),
      weights_(std::move(weights)),
      rope_cache_(std::move(rope_cache)) {}

Result<void> Qwen3Model::forward(const ForwardInput& input, ForwardOutput& output,
                                 DeviceBackend& backend, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    if (input.input_embeds_ == nullptr || output.logits_ == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    if (input.num_tokens_ <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    const int T = input.num_tokens_;
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

    if (nq % nkv != 0) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    if (rope_cache_.max_position() < T) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    if (weights_.layers_.size() < static_cast<size_t>(n_layers)) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    const int qkv_dim = (nq + 2 * nkv) * hd;
    const int attn_dim = nq * hd;

    auto pos_buf = backend.allocate_buffer(static_cast<size_t>(T) * sizeof(int32_t));
    {
        std::vector<int32_t> pos_host(static_cast<size_t>(T));
        for (int i = 0; i < T; ++i) pos_host[static_cast<size_t>(i)] = i;
        auto r = cuda_check(cudaMemcpy(pos_buf->data(), pos_host.data(),
                                       static_cast<size_t>(T) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice));
        if (!r) return std::unexpected(r.error());
    }

    auto hidden_a = backend.allocate_buffer(static_cast<size_t>(T) * D * sizeof(__nv_bfloat16));
    auto hidden_b = backend.allocate_buffer(static_cast<size_t>(T) * D * sizeof(__nv_bfloat16));
    auto normed = backend.allocate_buffer(static_cast<size_t>(T) * D * sizeof(__nv_bfloat16));
    auto qkv_out = backend.allocate_buffer(static_cast<size_t>(T) * qkv_dim * sizeof(__nv_bfloat16));
    auto q_buf = backend.allocate_buffer(static_cast<size_t>(T) * nq * hd * sizeof(__nv_bfloat16));
    auto k_buf = backend.allocate_buffer(static_cast<size_t>(T) * nkv * hd * sizeof(__nv_bfloat16));
    auto v_buf = backend.allocate_buffer(static_cast<size_t>(T) * nkv * hd * sizeof(__nv_bfloat16));
    auto attn_out = backend.allocate_buffer(static_cast<size_t>(T) * attn_dim * sizeof(__nv_bfloat16));
    auto gate = backend.allocate_buffer(static_cast<size_t>(T) * d_ff * sizeof(__nv_bfloat16));
    auto up = backend.allocate_buffer(static_cast<size_t>(T) * d_ff * sizeof(__nv_bfloat16));
    auto ffn_act = backend.allocate_buffer(static_cast<size_t>(T) * d_ff * sizeof(__nv_bfloat16));

    {
        auto r = cuda_check(cudaMemcpyAsync(
            hidden_a->data(), input.input_embeds_,
            static_cast<size_t>(T) * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, s));
        if (!r) return std::unexpected(r.error());
    }

    auto* hidden = static_cast<__nv_bfloat16*>(hidden_a->data());
    auto* next_hidden = static_cast<__nv_bfloat16*>(hidden_b->data());

    for (int l = 0; l < n_layers; ++l) {
        const Qwen3LayerWeights& lw = weights_.layers_[static_cast<size_t>(l)];
        Result<void> r{};

        // Attention: normed = RMSNorm(hidden)
        r = backend.rms_norm(RmsNormParams{
            .input_ = hidden, .weight_ = lw.rms_attn_->data(), .output_ = normed->data(),
            .rows_ = T, .dim_ = D, .eps_ = eps, .stream_ = s,
        });
        if (!r) return r;

        // qkv_out = normed @ qkv_weight^T
        r = backend.gemm(GemmParams{
            .a_ = lw.qkv_->data(), .b_ = normed->data(), .c_ = qkv_out->data(),
            .m_ = qkv_dim, .n_ = T, .k_ = D,
            .lda_ = D, .ldb_ = D, .ldc_ = qkv_dim,
            .trans_a_ = true, .trans_b_ = false, .stream_ = s,
        });
        if (!r) return r;

        r = backend.split_qkv(SplitQkvParams{
            .qkv_ = qkv_out->data(), .q_ = q_buf->data(), .k_ = k_buf->data(), .v_ = v_buf->data(),
            .num_tokens_ = T, .num_q_heads_ = nq, .num_kv_heads_ = nkv, .head_dim_ = hd, .stream_ = s,
        });
        if (!r) return r;

        // QK norm
        if (lw.q_norm_) {
            r = backend.rms_norm(RmsNormParams{
                .input_ = q_buf->data(), .weight_ = lw.q_norm_->data(), .output_ = q_buf->data(),
                .rows_ = T * nq, .dim_ = hd, .eps_ = eps, .stream_ = s,
            });
            if (!r) return r;
        }
        if (lw.k_norm_) {
            r = backend.rms_norm(RmsNormParams{
                .input_ = k_buf->data(), .weight_ = lw.k_norm_->data(), .output_ = k_buf->data(),
                .rows_ = T * nkv, .dim_ = hd, .eps_ = eps, .stream_ = s,
            });
            if (!r) return r;
        }

        // RoPE
        r = backend.rope(RopeParams{
            .q_ = q_buf->data(), .k_ = k_buf->data(),
            .positions_ = static_cast<const int32_t*>(pos_buf->data()),
            .rope_cache_ = rope_cache_.data(), .num_tokens_ = T, .num_q_heads_ = nq,
            .num_kv_heads_ = nkv, .head_dim_ = hd, .rotary_dim_ = hd,
            .max_position_ = rope_cache_.max_position(), .stream_ = s,
        });
        if (!r) return r;

        // Attention
        r = backend.naive_attention(NaiveAttnParams{
            .q_ = q_buf->data(), .k_ = k_buf->data(), .v_ = v_buf->data(), .output_ = attn_out->data(),
            .num_tokens_ = T, .num_q_heads_ = nq, .num_kv_heads_ = nkv, .head_dim_ = hd, .stream_ = s,
        });
        if (!r) return r;

        // O projection
        r = backend.gemm(GemmParams{
            .a_ = lw.o_->data(), .b_ = attn_out->data(), .c_ = next_hidden,
            .m_ = D, .n_ = T, .k_ = attn_dim,
            .lda_ = attn_dim, .ldb_ = attn_dim, .ldc_ = D,
            .trans_a_ = true, .trans_b_ = false, .stream_ = s,
        });
        if (!r) return r;

        r = backend.element_add(ElementAddParams{
            .dst_ = next_hidden, .src_ = hidden, .n_ = static_cast<int64_t>(T) * D, .stream_ = s,
        });
        if (!r) return r;
        std::swap(hidden, next_hidden);

        // FFN: normed = RMSNorm(hidden)
        r = backend.rms_norm(RmsNormParams{
            .input_ = hidden, .weight_ = lw.rms_ffn_->data(), .output_ = normed->data(),
            .rows_ = T, .dim_ = D, .eps_ = eps, .stream_ = s,
        });
        if (!r) return r;

        r = backend.gemm(GemmParams{
            .a_ = lw.gate_->data(), .b_ = normed->data(), .c_ = gate->data(),
            .m_ = d_ff, .n_ = T, .k_ = D,
            .lda_ = D, .ldb_ = D, .ldc_ = d_ff,
            .trans_a_ = true, .trans_b_ = false, .stream_ = s,
        });
        if (!r) return r;

        r = backend.gemm(GemmParams{
            .a_ = lw.up_->data(), .b_ = normed->data(), .c_ = up->data(),
            .m_ = d_ff, .n_ = T, .k_ = D,
            .lda_ = D, .ldb_ = D, .ldc_ = d_ff,
            .trans_a_ = true, .trans_b_ = false, .stream_ = s,
        });
        if (!r) return r;

        r = backend.silu_mul(SiluMulParams{
            .gate_ = gate->data(), .up_ = up->data(), .output_ = ffn_act->data(),
            .n_ = static_cast<int64_t>(T) * d_ff, .stream_ = s,
        });
        if (!r) return r;

        r = backend.gemm(GemmParams{
            .a_ = lw.down_->data(), .b_ = ffn_act->data(), .c_ = next_hidden,
            .m_ = D, .n_ = T, .k_ = d_ff,
            .lda_ = d_ff, .ldb_ = d_ff, .ldc_ = D,
            .trans_a_ = true, .trans_b_ = false, .stream_ = s,
        });
        if (!r) return r;

        r = backend.element_add(ElementAddParams{
            .dst_ = next_hidden, .src_ = hidden, .n_ = static_cast<int64_t>(T) * D, .stream_ = s,
        });
        if (!r) return r;
        std::swap(hidden, next_hidden);
    }

    auto r = backend.rms_norm(RmsNormParams{
        .input_ = hidden, .weight_ = weights_.rms_final_->data(), .output_ = normed->data(),
        .rows_ = T, .dim_ = D, .eps_ = eps, .stream_ = s,
    });
    if (!r) return r;

    r = backend.gemm(GemmParams{
        .a_ = weights_.lm_head_->data(), .b_ = normed->data(), .c_ = output.logits_,
        .m_ = V, .n_ = T, .k_ = D,
        .lda_ = D, .ldb_ = D, .ldc_ = V,
        .trans_a_ = true, .trans_b_ = false, .stream_ = s,
    });
    if (!r) return r;

    return {};
}

}  // namespace engine
}  // namespace ccinfer
