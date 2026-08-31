#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#include "common/error_code.h"
#include "facade/ops.h"

namespace ccinfer {

enum class ModelArch : uint8_t { Qwen3, Unknown };

struct ModelConfig {
    ModelArch arch_ = ModelArch::Unknown;

    int n_layers_ = 0;
    int n_q_heads_ = 0;
    int n_kv_heads_ = 0;
    int d_model_ = 0;
    int head_dim_ = 0;
    int d_ff_ = 0;
    int vocab_size_ = 0;

    int max_seq_len_ = 0;
    float rope_theta_ = 10000.0f;
    float rms_norm_eps_ = 1e-6f;

    ccop::DType weight_dtype_ = ccop::DType::kFloat16;

    const char* arch_name() const noexcept {
        switch (arch_) {
            case ModelArch::Qwen3:
                return "qwen3";
            default:
                return "unknown";
        }
    }

    static Result<ModelConfig> from_json(const nlohmann::json& j) {
        ModelConfig cfg;

        if (j.contains("architectures") && j["architectures"].is_array() &&
            !j["architectures"].empty() && j["architectures"][0].is_string()) {
            std::string arch = j["architectures"][0].get<std::string>();
            if (arch == "Qwen3ForCausalLM") {
                cfg.arch_ = ModelArch::Qwen3;
            }
        }
        if (cfg.arch_ == ModelArch::Unknown) {
            return std::unexpected(ErrorCode::ModelUnsupportedArch);
        }

        auto to_int = [](int64_t v) -> Result<int> {
            if (v < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
                v > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                return std::unexpected(ErrorCode::ModelConfigInvalid);
            }
            return static_cast<int>(v);
        };

        auto get_required_int = [&](const char* key) -> Result<int> {
            if (!j.contains(key) || !j[key].is_number_integer()) {
                return std::unexpected(ErrorCode::ModelConfigInvalid);
            }
            return to_int(j[key].get<int64_t>());
        };

        auto get_optional_int = [&](const char* key, int& out) -> Result<void> {
            if (!j.contains(key)) return {};
            if (!j[key].is_number_integer()) {
                return std::unexpected(ErrorCode::ModelConfigInvalid);
            }
            auto r = to_int(j[key].get<int64_t>());
            if (!r) return std::unexpected(r.error());
            out = *r;
            return {};
        };

        auto get_optional_float = [&](const char* key, float& out) -> Result<void> {
            if (!j.contains(key)) return {};
            if (!j[key].is_number()) {
                return std::unexpected(ErrorCode::ModelConfigInvalid);
            }
            out = j[key].get<float>();
            return {};
        };

        auto d_model = get_required_int("hidden_size");
        if (!d_model) return std::unexpected(d_model.error());
        cfg.d_model_ = *d_model;

        auto n_q_heads = get_required_int("num_attention_heads");
        if (!n_q_heads) return std::unexpected(n_q_heads.error());
        cfg.n_q_heads_ = *n_q_heads;

        auto n_layers = get_required_int("num_hidden_layers");
        if (!n_layers) return std::unexpected(n_layers.error());
        cfg.n_layers_ = *n_layers;

        auto d_ff = get_required_int("intermediate_size");
        if (!d_ff) return std::unexpected(d_ff.error());
        cfg.d_ff_ = *d_ff;

        auto vocab_size = get_required_int("vocab_size");
        if (!vocab_size) return std::unexpected(vocab_size.error());
        cfg.vocab_size_ = *vocab_size;

        if (cfg.d_model_ <= 0 || cfg.n_q_heads_ <= 0 || cfg.n_layers_ <= 0 || cfg.d_ff_ <= 0 ||
            cfg.vocab_size_ <= 0) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }

        if (cfg.d_model_ % cfg.n_q_heads_ != 0) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }
        int default_head_dim = cfg.d_model_ / cfg.n_q_heads_;
        cfg.n_kv_heads_ = cfg.n_q_heads_;
        if (auto r = get_optional_int("num_key_value_heads", cfg.n_kv_heads_); !r) {
            return std::unexpected(r.error());
        }

        cfg.head_dim_ = default_head_dim;
        if (auto r = get_optional_int("head_dim", cfg.head_dim_); !r) {
            return std::unexpected(r.error());
        }

        cfg.max_seq_len_ = 2048;
        if (auto r = get_optional_int("max_position_embeddings", cfg.max_seq_len_); !r) {
            return std::unexpected(r.error());
        }

        if (auto r = get_optional_float("rope_theta", cfg.rope_theta_); !r) {
            return std::unexpected(r.error());
        }
        if (auto r = get_optional_float("rms_norm_eps", cfg.rms_norm_eps_); !r) {
            return std::unexpected(r.error());
        }

        if (j.contains("torch_dtype") && j["torch_dtype"].is_string()) {
            const auto& dt = j["torch_dtype"].get<std::string>();
            if (dt == "bfloat16")
                cfg.weight_dtype_ = ccop::DType::kBFloat16;
            else if (dt == "float16")
                cfg.weight_dtype_ = ccop::DType::kFloat16;
            else if (dt == "float32")
                cfg.weight_dtype_ = ccop::DType::kFloat32;
            else
                return std::unexpected(ErrorCode::ModelUnsupportedDType);
        }

        if (!std::isfinite(cfg.rope_theta_) || !std::isfinite(cfg.rms_norm_eps_) ||
            cfg.rope_theta_ <= 0.0f || cfg.rms_norm_eps_ <= 0.0f) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }
        if (cfg.n_kv_heads_ <= 0 || cfg.max_seq_len_ <= 0 || cfg.head_dim_ <= 0) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }
        if (cfg.n_kv_heads_ > cfg.n_q_heads_ || cfg.n_q_heads_ % cfg.n_kv_heads_ != 0) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }
        // Qwen3 supports an explicit head_dim that is independent of d_model:
        // q/k/v projection dimensions are (n_q_heads * head_dim) and
        // (n_kv_heads * head_dim), which need not equal d_model.
        // Example: Qwen3-0.6B has hidden_size=1024, n_q_heads=16, head_dim=128.

        return cfg;
    }
};

}  // namespace ccinfer
