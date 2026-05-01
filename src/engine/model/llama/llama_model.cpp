#include "engine/model/llama/llama_model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "engine/backend/backend.h"
#include "engine/backend/cuda/cuda_utils.h"
#include "engine/core/device_buffer.h"
#include "engine/kernel/cuda_kernels.h"
#include "engine/model/registry.h"

namespace ccinfer {
namespace engine {

LlamaModel::LlamaModel(ModelConfig config, ModelWeights weights)
    : config_(std::move(config)), weights_(std::move(weights)) {
    rope_cache_.init(config_.max_seq_len_, config_.head_dim_, config_.rope_theta_);
}

Result<void> LlamaModel::forward(const ForwardInput& input, ForwardOutput& output,
                                 DeviceBackend& backend, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    if (input.input_embeds == nullptr || output.logits == nullptr) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    if (input.num_tokens <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    const int T = input.num_tokens;
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

    if (nq * hd != D) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
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

    // positions: [T]
    DeviceBuffer<int32_t> pos_buf(static_cast<size_t>(T));

    {
        std::vector<int32_t> pos_host(static_cast<size_t>(T));
        for (int i = 0; i < T; ++i) {
            pos_host[static_cast<size_t>(i)] = i;
        }

        auto r = cuda_check(cudaMemcpy(pos_buf.get(), pos_host.data(),
                                       static_cast<size_t>(T) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice));

        if (!r) {
            return std::unexpected(r.error());
        }
    }

    // Workspaces
    DeviceBuffer<__nv_bfloat16> hidden_a(static_cast<size_t>(T) * D);
    DeviceBuffer<__nv_bfloat16> hidden_b(static_cast<size_t>(T) * D);

    DeviceBuffer<__nv_bfloat16> normed(static_cast<size_t>(T) * D);

    DeviceBuffer<__nv_bfloat16> qkv_out(static_cast<size_t>(T) * qkv_dim);

    DeviceBuffer<__nv_bfloat16> q_buf(static_cast<size_t>(T) * nq * hd);
    DeviceBuffer<__nv_bfloat16> k_buf(static_cast<size_t>(T) * nkv * hd);
    DeviceBuffer<__nv_bfloat16> v_buf(static_cast<size_t>(T) * nkv * hd);

    DeviceBuffer<__nv_bfloat16> attn_out(static_cast<size_t>(T) * attn_dim);

    DeviceBuffer<__nv_bfloat16> gate(static_cast<size_t>(T) * d_ff);
    DeviceBuffer<__nv_bfloat16> up(static_cast<size_t>(T) * d_ff);
    DeviceBuffer<__nv_bfloat16> ffn_act(static_cast<size_t>(T) * d_ff);

    // hidden_a = input_embeds
    {
        auto r = cuda_check(cudaMemcpyAsync(
            hidden_a.get(), input.input_embeds,
            static_cast<size_t>(T) * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, s));

        if (!r) {
            return std::unexpected(r.error());
        }
    }

    __nv_bfloat16* hidden = hidden_a.get();
    __nv_bfloat16* next_hidden = hidden_b.get();

    // Transformer layers
    for (int l = 0; l < n_layers; ++l) {
        const LayerWeights& lw = weights_.layers_[static_cast<size_t>(l)];

        Result<void> r{};

        // Attention block: normed = RMSNorm(hidden)
        r = backend.rms_norm(RmsNormParams{
            .input_ = hidden,
            .weight_ = lw.rms_attn_.get(),
            .output_ = normed.get(),
            .rows_ = T,
            .dim_ = D,
            .eps_ = eps,
            .stream_ = s,
        });
        if (!r) {
            return r;
        }

        // qkv_out[T, qkv_dim] = normed[T, D] @ qkv_weight[qkv_dim, D]^T
        r = backend.gemm(GemmParams{
            .a_ = lw.qkv_.get(),
            .b_ = normed.get(),
            .c_ = qkv_out.get(),
            .m_ = qkv_dim,
            .n_ = T,
            .k_ = D,
            .lda_ = D,
            .ldb_ = D,
            .ldc_ = qkv_dim,
            .trans_a_ = true,
            .trans_b_ = false,
            .stream_ = s,
        });
        if (!r) {
            return r;
        }

        // qkv_out [T, Q|K|V] -> q/k/v token-major buffers
        r = launch_split_qkv(qkv_out.get(), q_buf.get(), k_buf.get(), v_buf.get(), T, nq, nkv, hd,
                             s);
        if (!r) {
            return r;
        }

        // Apply RoPE to Q and K in-place
        r = backend.rope(RopeParams{
            .q_ = q_buf.get(),
            .k_ = k_buf.get(),
            .positions_ = pos_buf.get(),
            .rope_cache_ = rope_cache_.data(),
            .num_tokens_ = T,
            .num_q_heads_ = nq,
            .num_kv_heads_ = nkv,
            .head_dim_ = hd,
            .rotary_dim_ = hd,
            .max_position_ = rope_cache_.max_position(),
            .stream_ = s,
        });
        if (!r) {
            return r;
        }

        // attn_out [T, nq, hd] = causal_attention(q, k, v)
        r = backend.naive_attention(NaiveAttnParams{
            .q_ = q_buf.get(),
            .k_ = k_buf.get(),
            .v_ = v_buf.get(),
            .output_ = attn_out.get(),
            .num_tokens_ = T,
            .num_q_heads_ = nq,
            .num_kv_heads_ = nkv,
            .head_dim_ = hd,
            .stream_ = s,
        });
        if (!r) {
            return r;
        }

        // next_hidden[T, D] = attn_out[T, D] @ o_proj[D, D]^T
        r = backend.gemm(GemmParams{
            .a_ = lw.o_.get(),
            .b_ = attn_out.get(),
            .c_ = next_hidden,
            .m_ = D,
            .n_ = T,
            .k_ = attn_dim,
            .lda_ = attn_dim,
            .ldb_ = attn_dim,
            .ldc_ = D,
            .trans_a_ = true,
            .trans_b_ = false,
            .stream_ = s,
        });
        if (!r) {
            return r;
        }

        // next_hidden += hidden
        r = launch_element_add(next_hidden, hidden, static_cast<int64_t>(T) * D, s);
        if (!r) {
            return r;
        }

        std::swap(hidden, next_hidden);

        // FFN block: normed = RMSNorm(hidden)
        r = backend.rms_norm(RmsNormParams{
            .input_ = hidden,
            .weight_ = lw.rms_ffn_.get(),
            .output_ = normed.get(),
            .rows_ = T,
            .dim_ = D,
            .eps_ = eps,
            .stream_ = s,
        });
        if (!r) {
            return r;
        }

        // gate[T, d_ff] = normed[T, D] @ gate_weight[d_ff, D]^T
        r = backend.gemm(GemmParams{
            .a_ = lw.gate_.get(),
            .b_ = normed.get(),
            .c_ = gate.get(),
            .m_ = d_ff,
            .n_ = T,
            .k_ = D,
            .lda_ = D,
            .ldb_ = D,
            .ldc_ = d_ff,
            .trans_a_ = true,
            .trans_b_ = false,
            .stream_ = s,
        });
        if (!r) {
            return r;
        }

        // up[T, d_ff] = normed[T, D] @ up_weight[d_ff, D]^T
        r = backend.gemm(GemmParams{
            .a_ = lw.up_.get(),
            .b_ = normed.get(),
            .c_ = up.get(),
            .m_ = d_ff,
            .n_ = T,
            .k_ = D,
            .lda_ = D,
            .ldb_ = D,
            .ldc_ = d_ff,
            .trans_a_ = true,
            .trans_b_ = false,
            .stream_ = s,
        });
        if (!r) {
            return r;
        }

        // ffn_act = SiLU(gate) * up
        r = backend.silu_mul(SiluMulParams{
            .gate_ = gate.get(),
            .up_ = up.get(),
            .output_ = ffn_act.get(),
            .n_ = static_cast<int64_t>(T) * d_ff,
            .stream_ = s,
        });
        if (!r) {
            return r;
        }

        // next_hidden[T, D] = ffn_act[T, d_ff] @ down_weight[D, d_ff]^T
        r = backend.gemm(GemmParams{
            .a_ = lw.down_.get(),
            .b_ = ffn_act.get(),
            .c_ = next_hidden,
            .m_ = D,
            .n_ = T,
            .k_ = d_ff,
            .lda_ = d_ff,
            .ldb_ = d_ff,
            .ldc_ = D,
            .trans_a_ = true,
            .trans_b_ = false,
            .stream_ = s,
        });
        if (!r) {
            return r;
        }

        // next_hidden += hidden
        r = launch_element_add(next_hidden, hidden, static_cast<int64_t>(T) * D, s);
        if (!r) {
            return r;
        }

        std::swap(hidden, next_hidden);
    }

    // Final RMSNorm
    auto r = backend.rms_norm(RmsNormParams{
        .input_ = hidden,
        .weight_ = weights_.rms_final_.get(),
        .output_ = normed.get(),
        .rows_ = T,
        .dim_ = D,
        .eps_ = eps,
        .stream_ = s,
    });
    if (!r) {
        return r;
    }

    // LM head: output_logits[T, V] = normed[T, D] @ lm_head[V, D]^T
    r = backend.gemm(GemmParams{
        .a_ = weights_.lm_head_.get(),
        .b_ = normed.get(),
        .c_ = output.logits,
        .m_ = V,
        .n_ = T,
        .k_ = D,
        .lda_ = D,
        .ldb_ = D,
        .ldc_ = V,
        .trans_a_ = true,
        .trans_b_ = false,
        .stream_ = s,
    });
    if (!r) {
        return r;
    }

    return {};
}

// Registration
namespace {

Result<std::unique_ptr<Model>> create_llama(ModelConfig config, ModelWeights weights) {
    return std::make_unique<LlamaModel>(std::move(config), std::move(weights));
}

const bool kLlamaRegistered = []() {
    ModelRegistry::instance().register_model("llama", create_llama);
    return true;
}();

}  // namespace

}  // namespace engine
}  // namespace ccinfer
