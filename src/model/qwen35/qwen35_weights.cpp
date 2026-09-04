#include "model/qwen35/qwen35_weights.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ccop/quant.h"
#include "model/qwen35/qwen35_model.h"
#include "quant/q8_0/q8_0_reference.h"

namespace ccinfer {

namespace {

std::uint16_t float_to_bfloat16(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);
    const std::uint32_t rounded = bits + rounding;
    return static_cast<std::uint16_t>(rounded >> 16);
}

std::int64_t shape_numel(const std::vector<std::int64_t>& shape) {
    std::int64_t n = 1;
    for (std::int64_t d : shape) n *= d;
    return n;
}

Result<std::vector<float>> read_q8_to_float(WeightSource& weights, const std::string& name,
                                            const std::vector<std::int64_t>& shape) {
    auto info_r = weights.info(name);
    if (!info_r) return std::unexpected(info_r.error());
    if (info_r->logical_shape != shape) return std::unexpected(ErrorCode::ModelShapeMismatch);
    const auto* qtype = std::get_if<ccop::QType>(&info_r->type);
    if (qtype == nullptr || qtype->quant_type != ccop::QuantType::kQ8_0) {
        return std::unexpected(ErrorCode::ModelUnsupportedDType);
    }

    const std::int64_t numel = shape_numel(shape);
    if (numel <= 0 || numel % ccop::kQ8_0BlockSize != 0) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }
    const std::size_t block_count = static_cast<std::size_t>(numel / ccop::kQ8_0BlockSize);

    auto raw_r = weights.read(name);
    if (!raw_r) return std::unexpected(raw_r.error());
    if (raw_r->size() != block_count * ccop::kQ8_0BlockBytes) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    std::vector<ccop::Q8_0Block> blocks(block_count);
    std::memcpy(blocks.data(), raw_r->data(), raw_r->size());
    std::vector<float> floats(static_cast<std::size_t>(numel));
    auto deq_r =
        dequantize_q8_0_reference(std::span<const ccop::Q8_0Block>(blocks.data(), block_count),
                                  std::span<float>(floats.data(), floats.size()));
    if (!deq_r) return std::unexpected(deq_r.error());
    return floats;
}

Result<std::vector<float>> read_f32_to_float(WeightSource& weights, const std::string& name,
                                             const std::vector<std::int64_t>& shape) {
    auto info_r = weights.info(name);
    if (!info_r) return std::unexpected(info_r.error());
    if (info_r->logical_shape != shape) return std::unexpected(ErrorCode::ModelShapeMismatch);
    const auto* dtype = std::get_if<ccop::DType>(&info_r->type);
    if (dtype == nullptr || *dtype != ccop::DType::kFloat32) {
        return std::unexpected(ErrorCode::ModelUnsupportedDType);
    }

    const std::int64_t numel = shape_numel(shape);
    if (numel <= 0) return std::unexpected(ErrorCode::ModelShapeMismatch);
    auto raw_r = weights.read(name);
    if (!raw_r) return std::unexpected(raw_r.error());
    if (raw_r->size() != static_cast<std::size_t>(numel) * sizeof(float)) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }
    std::vector<float> floats(static_cast<std::size_t>(numel));
    std::memcpy(floats.data(), raw_r->data(), raw_r->size());
    return floats;
}

std::vector<std::uint16_t> floats_to_bf16(const std::vector<float>& floats) {
    std::vector<std::uint16_t> out(floats.size());
    for (std::size_t i = 0; i < floats.size(); ++i) {
        out[i] = float_to_bfloat16(floats[i]);
    }
    return out;
}

Result<Tensor> load_q8_bf16(Backend& backend, WeightSource& weights, const std::string& name,
                            const std::vector<std::int64_t>& shape) {
    auto floats_r = read_q8_to_float(weights, name, shape);
    if (!floats_r) return std::unexpected(floats_r.error());
    auto bf16 = floats_to_bf16(*floats_r);
    return Tensor::from_host(backend, bf16.data(), ccop::DType::kBFloat16, shape);
}

Result<Tensor> load_f32_bf16(Backend& backend, WeightSource& weights, const std::string& name,
                             const std::vector<std::int64_t>& shape) {
    auto floats_r = read_f32_to_float(weights, name, shape);
    if (!floats_r) return std::unexpected(floats_r.error());
    auto bf16 = floats_to_bf16(*floats_r);
    return Tensor::from_host(backend, bf16.data(), ccop::DType::kBFloat16, shape);
}

Result<Tensor> load_f32(Backend& backend, WeightSource& weights, const std::string& name,
                        const std::vector<std::int64_t>& shape) {
    auto floats_r = read_f32_to_float(weights, name, shape);
    if (!floats_r) return std::unexpected(floats_r.error());
    return Tensor::from_host(backend, floats_r->data(), ccop::DType::kFloat32, shape);
}

Result<void> load_q_gate(Backend& backend, WeightSource& weights, int layer, int n_heads,
                         int head_dim, int d_model, Tensor& q_out, Tensor& gate_out) {
    const std::string name = "blk." + std::to_string(layer) + ".attn_q.weight";
    const std::vector<std::int64_t> shape{static_cast<std::int64_t>(n_heads * 2 * head_dim),
                                          d_model};
    auto floats_r = read_q8_to_float(weights, name, shape);
    if (!floats_r) return std::unexpected(floats_r.error());
    auto all_bf16 = floats_to_bf16(*floats_r);

    const std::size_t head_elems = static_cast<std::size_t>(head_dim * d_model);
    const std::size_t head_bytes = head_elems * sizeof(std::uint16_t);
    std::vector<std::uint16_t> q_bf16(head_elems * static_cast<std::size_t>(n_heads));
    std::vector<std::uint16_t> gate_bf16(head_elems * static_cast<std::size_t>(n_heads));

    for (int h = 0; h < n_heads; ++h) {
        const std::uint16_t* head_src =
            all_bf16.data() + static_cast<std::size_t>(h * 2 * head_dim * d_model);
        const std::uint16_t* q_src = head_src;
        const std::uint16_t* gate_src = head_src + head_elems;
        std::uint16_t* q_dst = q_bf16.data() + static_cast<std::size_t>(h) * head_elems;
        std::uint16_t* gate_dst = gate_bf16.data() + static_cast<std::size_t>(h) * head_elems;
        std::memcpy(q_dst, q_src, head_bytes);
        std::memcpy(gate_dst, gate_src, head_bytes);
    }

    const std::vector<std::int64_t> out_shape{static_cast<std::int64_t>(n_heads * head_dim),
                                              d_model};
    auto q_r = Tensor::from_host(backend, q_bf16.data(), ccop::DType::kBFloat16, out_shape);
    if (!q_r) return std::unexpected(q_r.error());
    auto gate_r = Tensor::from_host(backend, gate_bf16.data(), ccop::DType::kBFloat16, out_shape);
    if (!gate_r) return std::unexpected(gate_r.error());
    q_out = std::move(*q_r);
    gate_out = std::move(*gate_r);
    return {};
}

Result<Qwen35GdnLayerWeights> load_gdn_layer(Backend& backend, const ModelConfig& config,
                                             WeightSource& weights, int layer) {
    const int D = config.d_model_;
    const int d_ff = config.d_ff_;
    const int head_k_dim = config.ssm_state_size_;
    const int head_v_dim = config.ssm_inner_size_ / config.ssm_time_step_rank_;
    const int key_dim = config.ssm_group_count_ * head_k_dim;
    const int value_dim = config.ssm_time_step_rank_ * head_v_dim;
    const int conv_dim = 2 * key_dim + value_dim;
    const std::string p = "blk." + std::to_string(layer) + ".";

    Qwen35GdnLayerWeights w;
    auto assign_q8 = [&](Tensor& dst, const std::string& suffix,
                         const std::vector<std::int64_t>& shape) -> Result<void> {
        auto r = load_q8_bf16(backend, weights, p + suffix, shape);
        if (!r) return std::unexpected(r.error());
        dst = std::move(*r);
        return {};
    };
    auto assign_f32_bf16 = [&](Tensor& dst, const std::string& suffix,
                               const std::vector<std::int64_t>& shape) -> Result<void> {
        auto r = load_f32_bf16(backend, weights, p + suffix, shape);
        if (!r) return std::unexpected(r.error());
        dst = std::move(*r);
        return {};
    };
    auto assign_f32 = [&](Tensor& dst, const std::string& suffix,
                          const std::vector<std::int64_t>& shape) -> Result<void> {
        auto r = load_f32(backend, weights, p + suffix, shape);
        if (!r) return std::unexpected(r.error());
        dst = std::move(*r);
        return {};
    };

    if (auto r = assign_f32_bf16(w.attn_norm, "attn_norm.weight", {D}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.attn_qkv, "attn_qkv.weight", {conv_dim, D}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.attn_gate, "attn_gate.weight", {value_dim, D}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r =
            assign_f32_bf16(w.ssm_conv1d, "ssm_conv1d.weight", {conv_dim, config.ssm_conv_kernel_});
        !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.ssm_alpha, "ssm_alpha.weight", {config.ssm_time_step_rank_, D}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.ssm_beta, "ssm_beta.weight", {config.ssm_time_step_rank_, D}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.ssm_out, "ssm_out.weight", {D, value_dim}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_f32_bf16(w.ssm_norm, "ssm_norm.weight", {head_v_dim}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_f32(w.ssm_a, "ssm_a", {config.ssm_time_step_rank_}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_f32(w.ssm_dt_bias, "ssm_dt.bias", {config.ssm_time_step_rank_}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.ffn_gate, "ffn_gate.weight", {d_ff, D}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.ffn_up, "ffn_up.weight", {d_ff, D}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.ffn_down, "ffn_down.weight", {D, d_ff}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_f32_bf16(w.post_attention_norm, "post_attention_norm.weight", {D}); !r) {
        return std::unexpected(r.error());
    }
    return w;
}

Result<Qwen35AttnLayerWeights> load_attn_layer(Backend& backend, const ModelConfig& config,
                                               WeightSource& weights, int layer) {
    const int D = config.d_model_;
    const int d_ff = config.d_ff_;
    const int nq = config.n_q_heads_;
    const int nkv = config.n_kv_heads_;
    const int hd = config.head_dim_;
    const std::string p = "blk." + std::to_string(layer) + ".";

    Qwen35AttnLayerWeights w;
    auto assign_q8 = [&](Tensor& dst, const std::string& suffix,
                         const std::vector<std::int64_t>& shape) -> Result<void> {
        auto r = load_q8_bf16(backend, weights, p + suffix, shape);
        if (!r) return std::unexpected(r.error());
        dst = std::move(*r);
        return {};
    };
    auto assign_f32_bf16 = [&](Tensor& dst, const std::string& suffix,
                               const std::vector<std::int64_t>& shape) -> Result<void> {
        auto r = load_f32_bf16(backend, weights, p + suffix, shape);
        if (!r) return std::unexpected(r.error());
        dst = std::move(*r);
        return {};
    };

    if (auto r = assign_f32_bf16(w.attn_norm, "attn_norm.weight", {D}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = load_q_gate(backend, weights, layer, nq, hd, D, w.attn_q, w.attn_gate); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.attn_k, "attn_k.weight", {nkv * hd, D}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.attn_v, "attn_v.weight", {nkv * hd, D}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.attn_output, "attn_output.weight", {D, nq * hd}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_f32_bf16(w.attn_q_norm, "attn_q_norm.weight", {hd}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_f32_bf16(w.attn_k_norm, "attn_k_norm.weight", {hd}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.ffn_gate, "ffn_gate.weight", {d_ff, D}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.ffn_up, "ffn_up.weight", {d_ff, D}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_q8(w.ffn_down, "ffn_down.weight", {D, d_ff}); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = assign_f32_bf16(w.post_attention_norm, "post_attention_norm.weight", {D}); !r) {
        return std::unexpected(r.error());
    }
    return w;
}

}  // namespace

Result<Qwen35Weights> Qwen35Weights::load(Backend& backend, const ModelConfig& config,
                                          WeightSource& weights) {
    if (config.arch_ != ModelArch::Qwen3_5) {
        return std::unexpected(ErrorCode::ModelUnsupportedArch);
    }
    const int D = config.d_model_;
    const int vocab = config.vocab_size_;

    Qwen35Weights w;

    auto embed_r = load_q8_bf16(backend, weights, "token_embd.weight", {vocab, D});
    if (!embed_r) return std::unexpected(embed_r.error());
    w.embed = std::move(*embed_r);
    w.lm_head = w.embed;  // Tied embedding / lm_head.

    auto rms_final_r = load_f32_bf16(backend, weights, "output_norm.weight", {D});
    if (!rms_final_r) return std::unexpected(rms_final_r.error());
    w.rms_final = std::move(*rms_final_r);

    w.gdn_layers_.reserve(static_cast<std::size_t>(qwen35::num_gdn_layers(config)));
    w.attn_layers_.reserve(static_cast<std::size_t>(qwen35::num_kv_layers(config)));

    for (int layer = 0; layer < config.n_layers_; ++layer) {
        if (config.layer_types_[static_cast<std::size_t>(layer)] == LayerType::GatedDeltaNet) {
            auto layer_r = load_gdn_layer(backend, config, weights, layer);
            if (!layer_r) return std::unexpected(layer_r.error());
            w.gdn_layers_.push_back(std::move(*layer_r));
        } else {
            auto layer_r = load_attn_layer(backend, config, weights, layer);
            if (!layer_r) return std::unexpected(layer_r.error());
            w.attn_layers_.push_back(std::move(*layer_r));
        }
    }
    return w;
}

}  // namespace ccinfer
