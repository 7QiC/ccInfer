#include "engine/model/qwen3/qwen3_weights.h"

#include <cuda_runtime.h>

#include "engine/backend/cuda/cuda_utils.h"

namespace ccinfer {
namespace engine {

namespace {

void merge_qkv(DeviceBuffer<__nv_bfloat16>& qkv, DeviceBuffer<__nv_bfloat16>& q,
               DeviceBuffer<__nv_bfloat16>& k, DeviceBuffer<__nv_bfloat16>& v) {
    const size_t q_elems = q.size();
    const size_t k_elems = k.size();
    const size_t v_elems = v.size();
    const size_t total = q_elems + k_elems + v_elems;

    qkv = DeviceBuffer<__nv_bfloat16>(total);

    cudaMemcpy(qkv.get(), q.get(), q_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice);
    cudaMemcpy(qkv.get() + q_elems, k.get(), k_elems * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(qkv.get() + q_elems + k_elems, v.get(), v_elems * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToDevice);
}

}  // namespace

Result<Qwen3Weights> Qwen3Weights::load(const ModelConfig& config, const WeightLoader& loader) {
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

    Qwen3Weights w;

    w.embed_ = *loader.load<__nv_bfloat16>("model.embed_tokens.weight", {vocab, D});

    w.lm_head_ = *loader.load<__nv_bfloat16>("lm_head.weight", {vocab, D});

    w.rms_final_ = *loader.load<__nv_bfloat16>("model.norm.weight", {D});

    w.layers_.reserve(static_cast<size_t>(n_layers));

    for (int i = 0; i < n_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i);

        Qwen3LayerWeights lw;

        lw.o_ = *loader.load<__nv_bfloat16>(p + ".self_attn.o_proj.weight", {D, nq * hd});
        lw.gate_ = *loader.load<__nv_bfloat16>(p + ".mlp.gate_proj.weight", {d_ff, D});
        lw.up_ = *loader.load<__nv_bfloat16>(p + ".mlp.up_proj.weight", {d_ff, D});
        lw.down_ = *loader.load<__nv_bfloat16>(p + ".mlp.down_proj.weight", {D, d_ff});
        lw.rms_attn_ = *loader.load<__nv_bfloat16>(p + ".input_layernorm.weight", {D});
        lw.rms_ffn_ = *loader.load<__nv_bfloat16>(p + ".post_attention_layernorm.weight", {D});

        auto q = loader.load<__nv_bfloat16>(p + ".self_attn.q_proj.weight", {nq * hd, D});
        auto k = loader.load<__nv_bfloat16>(p + ".self_attn.k_proj.weight", {nkv * hd, D});
        auto v = loader.load<__nv_bfloat16>(p + ".self_attn.v_proj.weight", {nkv * hd, D});

        merge_qkv(lw.qkv_, *q, *k, *v);

        // QK norm (Qwen3-specific).
        auto qn = loader.load<__nv_bfloat16>(p + ".self_attn.q_norm.weight", {hd});
        if (qn) lw.q_norm_ = std::move(*qn);

        auto kn = loader.load<__nv_bfloat16>(p + ".self_attn.k_norm.weight", {hd});
        if (kn) lw.k_norm_ = std::move(*kn);

        w.layers_.push_back(std::move(lw));
    }

    return w;
}

}  // namespace engine
}  // namespace ccinfer
