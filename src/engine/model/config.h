#pragma once

#include <string>
#include <nlohmann/json.hpp>

#include "dtype.h"
#include "result.h"

namespace ccinfer {

enum class ModelArch : uint8_t { Llama, Unknown };

struct ModelConfig {
    ModelArch arch = ModelArch::Unknown;

    int n_layers   = 0;
    int n_q_heads  = 0;
    int n_kv_heads = 0;
    int d_model    = 0;
    int head_dim   = 0;
    int d_ff       = 0;
    int vocab_size = 0;

    int max_seq_len    = 0;
    float rope_theta   = 10000.0f;
    float rms_norm_eps = 1e-6f;

    DType weight_dtype = DType::kFloat16;

    static Result<ModelConfig> from_json(const nlohmann::json& j) {
        ModelConfig cfg;

        if (j.contains("architectures") && j["architectures"].is_array() &&
            !j["architectures"].empty()) {
            std::string arch = j["architectures"][0].get<std::string>();
            if (arch.find("Llama") != std::string::npos) {
                cfg.arch = ModelArch::Llama;
            } else {
                cfg.arch = ModelArch::Unknown;
            }
        }

        if (!j.contains("hidden_size") || !j.contains("num_attention_heads") ||
            !j.contains("num_hidden_layers") || !j.contains("intermediate_size") ||
            !j.contains("vocab_size")) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        cfg.d_model    = j["hidden_size"].get<int>();
        cfg.n_q_heads  = j["num_attention_heads"].get<int>();
        cfg.n_layers   = j["num_hidden_layers"].get<int>();
        cfg.d_ff       = j["intermediate_size"].get<int>();
        cfg.vocab_size = j["vocab_size"].get<int>();

        cfg.n_kv_heads = j.value("num_key_value_heads", cfg.n_q_heads);
        cfg.head_dim   = j.value("head_dim", cfg.d_model / cfg.n_q_heads);
        cfg.max_seq_len = j.value("max_position_embeddings", 2048);
        cfg.rope_theta  = j.value("rope_theta", 10000.0f);
        cfg.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);

        return cfg;
    }
};

} // namespace ccinfer
