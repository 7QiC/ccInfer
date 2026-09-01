#include "checkpoint/huggingface/config_loader.h"

#include <fstream>
#include <limits>
#include <string>

namespace ccinfer {

namespace {

Result<ModelConfig> from_json_impl(const nlohmann::json& j) {
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

    // torch_dtype is intentionally not stored in ModelConfig:
    // ModelConfig only describes architecture. Weight storage and execution
    // dtype are represented independently by Weight / runtime precision.

    auto validation = cfg.validate();
    if (!validation) return std::unexpected(validation.error());
    return cfg;
}

}  // namespace

Result<ModelConfig> HfConfigLoader::load(const std::string& config_json_path) {
    std::ifstream cfg_file(config_json_path);
    if (!cfg_file.is_open()) return std::unexpected(ErrorCode::ModelLoadFailed);

    auto json = nlohmann::json::parse(cfg_file, nullptr, false);
    if (json.is_discarded()) return std::unexpected(ErrorCode::ModelConfigInvalid);
    return load_from_json(json);
}

Result<ModelConfig> HfConfigLoader::load_from_json(const nlohmann::json& j) {
    return from_json_impl(j);
}

}  // namespace ccinfer
