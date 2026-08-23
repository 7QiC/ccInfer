#include "model/qwen3/qwen3_weights.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ccinfer {

namespace {

Result<void> merge_qkv(Tensor& qkv, const Tensor& q, const Tensor& k, const Tensor& v, int qkv_dim,
                       int d_model, Backend& backend) {
    const std::size_t q_bytes = q.nbytes();
    const std::size_t k_bytes = k.nbytes();
    const std::size_t v_bytes = v.nbytes();
    const std::size_t kMax = std::numeric_limits<std::size_t>::max();
    if (q_bytes > kMax - k_bytes || v_bytes > kMax - q_bytes - k_bytes) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    const std::size_t total_bytes = q_bytes + k_bytes + v_bytes;

    auto qkv_r = backend.allocate_buffer(total_bytes);
    if (!qkv_r) return std::unexpected(qkv_r.error());
    auto qkv_buffer = std::move(*qkv_r);

    auto* dst = static_cast<__nv_bfloat16*>(qkv_buffer->data());
    auto r = backend.memcpy_d2d(dst, q.data(), q_bytes);
    if (!r) return r;
    r = backend.memcpy_d2d(dst + q_bytes / sizeof(__nv_bfloat16), k.data(), k_bytes);
    if (!r) return r;
    r = backend.memcpy_d2d(dst + (q_bytes + k_bytes) / sizeof(__nv_bfloat16), v.data(), v_bytes);
    if (!r) return r;

    auto sync_r = backend.synchronize();
    if (!sync_r) return sync_r;
    qkv = Tensor(std::move(qkv_buffer), ccop::DType::kBFloat16, {qkv_dim, d_model});
    return {};
}

}  // namespace

Result<Qwen3Weights> Qwen3Weights::load(Backend& backend, const ModelConfig& config,
                                        const WeightLoader& loader) {
    if (config.weight_dtype_ != ccop::DType::kBFloat16) {
        return std::unexpected(ErrorCode::ModelUnsupportedDType);
    }

    if (config.d_model_ <= 0 || config.n_q_heads_ <= 0 || config.n_kv_heads_ <= 0 ||
        config.head_dim_ <= 0 || config.n_layers_ <= 0 || config.d_ff_ <= 0 ||
        config.vocab_size_ <= 0) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    if (config.n_q_heads_ % config.n_kv_heads_ != 0) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    const int D = config.d_model_;
    const int nq = config.n_q_heads_;
    const int nkv = config.n_kv_heads_;
    const int hd = config.head_dim_;
    const int n_layers = config.n_layers_;
    const int d_ff = config.d_ff_;
    const int vocab = config.vocab_size_;

    const int qkv_dim = (nq + 2 * nkv) * hd;

    Qwen3Weights w;

    {
        auto r =
            loader.load_tensor<__nv_bfloat16>(backend, "model.embed_tokens.weight", {vocab, D});
        if (!r) return std::unexpected(r.error());
        w.embed = std::move(*r);
    }

    {
        auto r = loader.load_tensor<__nv_bfloat16>(backend, "lm_head.weight", {vocab, D});
        if (!r) return std::unexpected(r.error());
        w.lm_head = std::move(*r);
    }

    {
        auto r = loader.load_tensor<__nv_bfloat16>(backend, "model.norm.weight", {D});
        if (!r) return std::unexpected(r.error());
        w.rms_final = std::move(*r);
    }

    w.layers_.reserve(static_cast<std::size_t>(n_layers));

    for (int i = 0; i < n_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i);

        Qwen3LayerWeights lw;

        {
            auto r = loader.load_tensor<__nv_bfloat16>(backend, p + ".self_attn.o_proj.weight",
                                                       {D, nq * hd});
            if (!r) return std::unexpected(r.error());
            lw.o = std::move(*r);
        }
        {
            auto r =
                loader.load_tensor<__nv_bfloat16>(backend, p + ".mlp.gate_proj.weight", {d_ff, D});
            if (!r) return std::unexpected(r.error());
            lw.gate = std::move(*r);
        }
        {
            auto r =
                loader.load_tensor<__nv_bfloat16>(backend, p + ".mlp.up_proj.weight", {d_ff, D});
            if (!r) return std::unexpected(r.error());
            lw.up = std::move(*r);
        }
        {
            auto r =
                loader.load_tensor<__nv_bfloat16>(backend, p + ".mlp.down_proj.weight", {D, d_ff});
            if (!r) return std::unexpected(r.error());
            lw.down = std::move(*r);
        }
        {
            auto r = loader.load_tensor<__nv_bfloat16>(backend, p + ".input_layernorm.weight", {D});
            if (!r) return std::unexpected(r.error());
            lw.rms_attn = std::move(*r);
        }
        {
            auto r = loader.load_tensor<__nv_bfloat16>(backend,
                                                       p + ".post_attention_layernorm.weight", {D});
            if (!r) return std::unexpected(r.error());
            lw.rms_ffn = std::move(*r);
        }

        auto q = loader.load_tensor<__nv_bfloat16>(backend, p + ".self_attn.q_proj.weight",
                                                   {nq * hd, D});
        if (!q) return std::unexpected(q.error());
        auto k = loader.load_tensor<__nv_bfloat16>(backend, p + ".self_attn.k_proj.weight",
                                                   {nkv * hd, D});
        if (!k) return std::unexpected(k.error());
        auto v = loader.load_tensor<__nv_bfloat16>(backend, p + ".self_attn.v_proj.weight",
                                                   {nkv * hd, D});
        if (!v) return std::unexpected(v.error());

        auto mqkv = merge_qkv(lw.qkv, *q, *k, *v, qkv_dim, D, backend);
        if (!mqkv) return std::unexpected(mqkv.error());

        // QK norm (Qwen3-specific).
        if (loader.has(p + ".self_attn.q_norm.weight")) {
            auto qn =
                loader.load_tensor<__nv_bfloat16>(backend, p + ".self_attn.q_norm.weight", {hd});
            if (!qn) return std::unexpected(qn.error());
            lw.q_norm = std::move(*qn);
        }

        if (loader.has(p + ".self_attn.k_norm.weight")) {
            auto kn =
                loader.load_tensor<__nv_bfloat16>(backend, p + ".self_attn.k_norm.weight", {hd});
            if (!kn) return std::unexpected(kn.error());
            lw.k_norm = std::move(*kn);
        }

        w.layers_.push_back(std::move(lw));
    }

    return w;
}

}  // namespace ccinfer
