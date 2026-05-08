#include "engine/model/llama2/llama2_weights.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>

namespace ccinfer {
namespace engine {

namespace {

void merge_qkv(std::unique_ptr<DeviceBuffer>& qkv,
               std::unique_ptr<DeviceBuffer>& q,
               std::unique_ptr<DeviceBuffer>& k,
               std::unique_ptr<DeviceBuffer>& v,
               DefaultBackend& backend) {
    const size_t q_bytes = q->bytes();
    const size_t k_bytes = k->bytes();
    const size_t v_bytes = v->bytes();
    const size_t total_bytes = q_bytes + k_bytes + v_bytes;

    qkv = backend.allocate_buffer(total_bytes);

    auto* dst = static_cast<__nv_bfloat16*>(qkv->data());
    backend.memcpy_d2d(dst, static_cast<const __nv_bfloat16*>(q->data()), q_bytes);
    backend.memcpy_d2d(dst + q_bytes / sizeof(__nv_bfloat16),
                       static_cast<const __nv_bfloat16*>(k->data()), k_bytes);
    backend.memcpy_d2d(dst + (q_bytes + k_bytes) / sizeof(__nv_bfloat16),
                       static_cast<const __nv_bfloat16*>(v->data()), v_bytes);
}

}  // namespace

Result<LlamaWeights> LlamaWeights::load(DefaultBackend& backend,
                                        const ModelConfig& config,
                                        const WeightLoader& loader) {
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

    LlamaWeights w;

    {
        auto r = loader.load<__nv_bfloat16>(backend, "model.embed_tokens.weight", {vocab, D});
        if (!r) return std::unexpected(r.error());
        w.embed_ = std::move(*r);
    }

    {
        auto r = loader.load<__nv_bfloat16>(backend, "lm_head.weight", {vocab, D});
        if (!r) return std::unexpected(r.error());
        w.lm_head_ = std::move(*r);
    }

    {
        auto r = loader.load<__nv_bfloat16>(backend, "model.norm.weight", {D});
        if (!r) return std::unexpected(r.error());
        w.rms_final_ = std::move(*r);
    }

    w.layers_.reserve(static_cast<size_t>(n_layers));

    for (int i = 0; i < n_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i);

        LlamaLayerWeights lw;

        {
            auto r = loader.load<__nv_bfloat16>(backend, p + ".self_attn.o_proj.weight",
                                                {D, nq * hd});
            if (!r) return std::unexpected(r.error());
            lw.o_ = std::move(*r);
        }
        {
            auto r = loader.load<__nv_bfloat16>(backend, p + ".mlp.gate_proj.weight", {d_ff, D});
            if (!r) return std::unexpected(r.error());
            lw.gate_ = std::move(*r);
        }
        {
            auto r = loader.load<__nv_bfloat16>(backend, p + ".mlp.up_proj.weight", {d_ff, D});
            if (!r) return std::unexpected(r.error());
            lw.up_ = std::move(*r);
        }
        {
            auto r = loader.load<__nv_bfloat16>(backend, p + ".mlp.down_proj.weight", {D, d_ff});
            if (!r) return std::unexpected(r.error());
            lw.down_ = std::move(*r);
        }
        {
            auto r = loader.load<__nv_bfloat16>(backend, p + ".input_layernorm.weight", {D});
            if (!r) return std::unexpected(r.error());
            lw.rms_attn_ = std::move(*r);
        }
        {
            auto r = loader.load<__nv_bfloat16>(backend, p + ".post_attention_layernorm.weight",
                                                {D});
            if (!r) return std::unexpected(r.error());
            lw.rms_ffn_ = std::move(*r);
        }

        auto q = loader.load<__nv_bfloat16>(backend, p + ".self_attn.q_proj.weight",
                                            {nq * hd, D});
        auto k = loader.load<__nv_bfloat16>(backend, p + ".self_attn.k_proj.weight",
                                            {nkv * hd, D});
        auto v = loader.load<__nv_bfloat16>(backend, p + ".self_attn.v_proj.weight",
                                            {nkv * hd, D});
        if (!q || !k || !v) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        merge_qkv(lw.qkv_, *q, *k, *v, backend);

        w.layers_.push_back(std::move(lw));
    }

    return w;
}

}  // namespace engine
}  // namespace ccinfer
