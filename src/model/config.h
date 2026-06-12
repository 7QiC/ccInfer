#pragma once

#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>

#include "base/result.h"
#include "core/dtype.h"

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

    DType weight_dtype_ = DType::kFloat16;

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
            if (arch.find("Qwen") != std::string::npos) {
                cfg.arch_ = ModelArch::Qwen3;
            }
        }
        if (cfg.arch_ == ModelArch::Unknown) {
            return std::unexpected(ErrorCode::ModelUnsupportedArch);
        }

        if (!j.contains("hidden_size") || !j["hidden_size"].is_number_integer() ||
            !j.contains("num_attention_heads") || !j["num_attention_heads"].is_number_integer() ||
            !j.contains("num_hidden_layers") || !j["num_hidden_layers"].is_number_integer() ||
            !j.contains("intermediate_size") || !j["intermediate_size"].is_number_integer() ||
            !j.contains("vocab_size") || !j["vocab_size"].is_number_integer()) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }

        auto to_int = [](int64_t v) -> Result<int> {
            if (v < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
                v > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                return std::unexpected(ErrorCode::ModelConfigInvalid);
            }
            return static_cast<int>(v);
        };

        auto r = to_int(j["hidden_size"].get<int64_t>());
        if (!r) return std::unexpected(r.error());
        cfg.d_model_ = *r;

        r = to_int(j["num_attention_heads"].get<int64_t>());
        if (!r) return std::unexpected(r.error());
        cfg.n_q_heads_ = *r;

        r = to_int(j["num_hidden_layers"].get<int64_t>());
        if (!r) return std::unexpected(r.error());
        cfg.n_layers_ = *r;

        r = to_int(j["intermediate_size"].get<int64_t>());
        if (!r) return std::unexpected(r.error());
        cfg.d_ff_ = *r;

        r = to_int(j["vocab_size"].get<int64_t>());
        if (!r) return std::unexpected(r.error());
        cfg.vocab_size_ = *r;

        if (cfg.d_model_ <= 0 || cfg.n_q_heads_ <= 0 || cfg.n_layers_ <= 0 || cfg.d_ff_ <= 0 ||
            cfg.vocab_size_ <= 0) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }

        int default_head_dim = cfg.d_model_ / cfg.n_q_heads_;
        cfg.n_kv_heads_ = cfg.n_q_heads_;
        if (j.contains("num_key_value_heads")) {
            if (!j["num_key_value_heads"].is_number_integer())
                return std::unexpected(ErrorCode::ModelConfigInvalid);
            auto r = to_int(j["num_key_value_heads"].get<int64_t>());
            if (!r) return std::unexpected(r.error());
            cfg.n_kv_heads_ = *r;
        }

        cfg.head_dim_ = default_head_dim;
        if (j.contains("head_dim")) {
            if (!j["head_dim"].is_number_integer())
                return std::unexpected(ErrorCode::ModelConfigInvalid);
            auto r = to_int(j["head_dim"].get<int64_t>());
            if (!r) return std::unexpected(r.error());
            cfg.head_dim_ = *r;
        }

        cfg.max_seq_len_ = 2048;
        if (j.contains("max_position_embeddings")) {
            if (!j["max_position_embeddings"].is_number_integer())
                return std::unexpected(ErrorCode::ModelConfigInvalid);
            auto r = to_int(j["max_position_embeddings"].get<int64_t>());
            if (!r) return std::unexpected(r.error());
            cfg.max_seq_len_ = *r;
        }

        if (j.contains("rope_theta")) {
            if (!j["rope_theta"].is_number()) return std::unexpected(ErrorCode::ModelConfigInvalid);
            cfg.rope_theta_ = j["rope_theta"].get<float>();
        }
        if (j.contains("rms_norm_eps")) {
            if (!j["rms_norm_eps"].is_number())
                return std::unexpected(ErrorCode::ModelConfigInvalid);
            cfg.rms_norm_eps_ = j["rms_norm_eps"].get<float>();
        }

        if (j.contains("torch_dtype") && j["torch_dtype"].is_string()) {
            const auto& dt = j["torch_dtype"].get<std::string>();
            if (dt == "bfloat16")
                cfg.weight_dtype_ = DType::kBFloat16;
            else if (dt == "float16")
                cfg.weight_dtype_ = DType::kFloat16;
            else if (dt == "float32")
                cfg.weight_dtype_ = DType::kFloat32;
            else
                return std::unexpected(ErrorCode::ModelUnsupportedDType);
        }

        if (cfg.rope_theta_ <= 0.0f || cfg.rms_norm_eps_ <= 0.0f) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }
        if (cfg.n_kv_heads_ <= 0 || cfg.max_seq_len_ <= 0 || cfg.head_dim_ <= 0) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }
        if (cfg.n_q_heads_ % cfg.n_kv_heads_ != 0) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }

        return cfg;
    }
};

}  // namespace ccinfer
