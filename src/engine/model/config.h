#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>

#include "common/dtype.h"
#include "common/result.h"

namespace ccinfer {
namespace engine {

enum class ModelArch : uint8_t { Llama, Unknown };

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

    static Result<ModelConfig> from_json(const nlohmann::json& j) {
        ModelConfig cfg;

        if (j.contains("architectures") && j["architectures"].is_array() &&
            !j["architectures"].empty()) {
            std::string arch = j["architectures"][0].get<std::string>();
            if (arch.find("Llama") != std::string::npos) {
                cfg.arch_ = ModelArch::Llama;
            } else {
                cfg.arch_ = ModelArch::Unknown;
            }
        }

        if (!j.contains("hidden_size") || !j.contains("num_attention_heads") ||
            !j.contains("num_hidden_layers") || !j.contains("intermediate_size") ||
            !j.contains("vocab_size")) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        cfg.d_model_ = j["hidden_size"].get<int>();
        cfg.n_q_heads_ = j["num_attention_heads"].get<int>();
        cfg.n_layers_ = j["num_hidden_layers"].get<int>();
        cfg.d_ff_ = j["intermediate_size"].get<int>();
        cfg.vocab_size_ = j["vocab_size"].get<int>();

        cfg.n_kv_heads_ = j.value("num_key_value_heads", cfg.n_q_heads_);
        cfg.head_dim_ = j.value("head_dim", cfg.d_model_ / cfg.n_q_heads_);
        cfg.max_seq_len_ = j.value("max_position_embeddings", 2048);
        cfg.rope_theta_ = j.value("rope_theta", 10000.0f);
        cfg.rms_norm_eps_ = j.value("rms_norm_eps", 1e-6f);

        return cfg;
    }
};

}  // namespace engine
}  // namespace ccinfer
