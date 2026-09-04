#include "model/qwen3/qwen3_weights.h"

#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ccinfer {

namespace {

Result<void> merge_qkv(Tensor& qkv, const Tensor& q, const Tensor& k, const Tensor& v, int qkv_dim,
                       int d_model, Backend& backend) {
    const std::size_t q_bytes = q.nbytes();
    const std::size_t k_bytes = k.nbytes();
    const std::size_t v_bytes = v.nbytes();
    const std::size_t kMax = std::numeric_limits<std::size_t>::max();
    assert(q_bytes <= kMax - k_bytes);
    assert(v_bytes <= kMax - q_bytes - k_bytes);
    const std::size_t total_bytes = q_bytes + k_bytes + v_bytes;

    auto qkv_r = backend.allocate_buffer(total_bytes);
    if (!qkv_r) return std::unexpected(qkv_r.error());
    auto qkv_buffer = std::move(*qkv_r);

    auto* dst = static_cast<char*>(qkv_buffer->data());
    auto r = backend.memcpy_d2d(dst, q.data(), q_bytes);
    if (!r) return r;
    r = backend.memcpy_d2d(dst + q_bytes, k.data(), k_bytes);
    if (!r) return r;
    r = backend.memcpy_d2d(dst + q_bytes + k_bytes, v.data(), v_bytes);
    if (!r) return r;

    auto sync_r = backend.synchronize();
    if (!sync_r) return sync_r;
    qkv = Tensor(std::move(qkv_buffer), ccop::DType::kBFloat16, {qkv_dim, d_model});
    return {};
}

}  // namespace

Result<Qwen3Weights> Qwen3Weights::load(Backend& backend, const ModelConfig& config,
                                        WeightSource& weights) {
    const int D = config.d_model_;
    const int nq = config.n_q_heads_;
    const int nkv = config.n_kv_heads_;
    const int hd = config.head_dim_;
    const int n_layers = config.n_layers_;
    const int d_ff = config.d_ff_;
    const int vocab = config.vocab_size_;

    const int qkv_dim = (nq + 2 * nkv) * hd;

    Qwen3Weights w;

    auto load_tensor = [&](const std::string& name, std::vector<int64_t> shape) -> Result<Tensor> {
        auto info_r = weights.info(name);
        if (!info_r) return std::unexpected(info_r.error());
        const auto& info = *info_r;
        if (info.logical_shape != shape) return std::unexpected(ErrorCode::ModelShapeMismatch);

        const auto* dtype = std::get_if<ccop::DType>(&info.storage_type);
        if (dtype == nullptr || *dtype != ccop::DType::kBFloat16) {
            return std::unexpected(ErrorCode::ModelUnsupportedDType);
        }

        auto raw = weights.read(name);
        if (!raw) return std::unexpected(raw.error());
        return Tensor::from_host(backend, raw->data(), *dtype, shape);
    };

    auto embed = load_tensor("model.embed_tokens.weight", {vocab, D});
    if (!embed) return std::unexpected(embed.error());
    w.embed = std::move(*embed);

    auto lm_head = load_tensor("lm_head.weight", {vocab, D});
    if (!lm_head) return std::unexpected(lm_head.error());
    w.lm_head = std::move(*lm_head);

    auto rms_final = load_tensor("model.norm.weight", {D});
    if (!rms_final) return std::unexpected(rms_final.error());
    w.rms_final = std::move(*rms_final);

    w.layers_.reserve(static_cast<std::size_t>(n_layers));

    for (int i = 0; i < n_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i);

        Qwen3LayerWeights lw;

        auto o = load_tensor(p + ".self_attn.o_proj.weight", {D, nq * hd});
        if (!o) return std::unexpected(o.error());
        lw.o = std::move(*o);

        auto gate = load_tensor(p + ".mlp.gate_proj.weight", {d_ff, D});
        if (!gate) return std::unexpected(gate.error());
        lw.gate = std::move(*gate);

        auto up = load_tensor(p + ".mlp.up_proj.weight", {d_ff, D});
        if (!up) return std::unexpected(up.error());
        lw.up = std::move(*up);

        auto down = load_tensor(p + ".mlp.down_proj.weight", {D, d_ff});
        if (!down) return std::unexpected(down.error());
        lw.down = std::move(*down);

        auto rms_attn = load_tensor(p + ".input_layernorm.weight", {D});
        if (!rms_attn) return std::unexpected(rms_attn.error());
        lw.rms_attn = std::move(*rms_attn);

        auto rms_ffn = load_tensor(p + ".post_attention_layernorm.weight", {D});
        if (!rms_ffn) return std::unexpected(rms_ffn.error());
        lw.rms_ffn = std::move(*rms_ffn);

        auto q = load_tensor(p + ".self_attn.q_proj.weight", {nq * hd, D});
        if (!q) return std::unexpected(q.error());
        auto k = load_tensor(p + ".self_attn.k_proj.weight", {nkv * hd, D});
        if (!k) return std::unexpected(k.error());
        auto v = load_tensor(p + ".self_attn.v_proj.weight", {nkv * hd, D});
        if (!v) return std::unexpected(v.error());

        auto mqkv = merge_qkv(lw.qkv, *q, *k, *v, qkv_dim, D, backend);
        if (!mqkv) return std::unexpected(mqkv.error());

        // QK norm (Qwen3-specific).
        if (weights.has(p + ".self_attn.q_norm.weight")) {
            auto qn = load_tensor(p + ".self_attn.q_norm.weight", {hd});
            if (!qn) return std::unexpected(qn.error());
            lw.q_norm = std::move(*qn);
        }

        if (weights.has(p + ".self_attn.k_norm.weight")) {
            auto kn = load_tensor(p + ".self_attn.k_norm.weight", {hd});
            if (!kn) return std::unexpected(kn.error());
            lw.k_norm = std::move(*kn);
        }

        w.layers_.push_back(std::move(lw));
    }

    return w;
}

}  // namespace ccinfer
