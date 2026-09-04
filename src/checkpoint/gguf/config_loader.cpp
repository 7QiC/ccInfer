#include "checkpoint/gguf/config_loader.h"

#include <cstdint>
#include <optional>
#include <string>

namespace ccinfer {

namespace {

std::optional<uint64_t> metadata_u64(const GGUFReader& reader, const std::string& key) {
    auto v = reader.metadata(key);
    if (!v.has_value()) return std::nullopt;
    return v->as_u64();
}

std::optional<float> metadata_f32(const GGUFReader& reader, const std::string& key) {
    auto v = reader.metadata(key);
    if (!v.has_value()) return std::nullopt;
    return v->as_f32();
}

std::optional<int> metadata_int(const GGUFReader& reader, const std::string& key) {
    auto v = metadata_u64(reader, key);
    if (!v) return std::nullopt;
    if (*v > static_cast<uint64_t>(INT32_MAX)) return std::nullopt;
    return static_cast<int>(*v);
}

}  // namespace

Result<ModelConfig> GgufConfigLoader::load(const GGUFReader& reader) {
    ModelConfig cfg;

    auto arch = reader.metadata("general.architecture");
    if (!arch.has_value() || arch->as_string() != "qwen35") {
        return std::unexpected(ErrorCode::ModelUnsupportedArch);
    }
    cfg.arch_ = ModelArch::Qwen3_5;

    auto get_int = [&](const std::string& key) -> Result<int> {
        auto v = metadata_int(reader, key);
        if (!v.has_value()) return std::unexpected(ErrorCode::ModelConfigInvalid);
        return *v;
    };

    auto block_count = get_int("qwen35.block_count");
    if (!block_count) return std::unexpected(block_count.error());
    auto embedding_length = get_int("qwen35.embedding_length");
    if (!embedding_length) return std::unexpected(embedding_length.error());
    auto feed_forward_length = get_int("qwen35.feed_forward_length");
    if (!feed_forward_length) return std::unexpected(feed_forward_length.error());
    auto head_count = get_int("qwen35.attention.head_count");
    if (!head_count) return std::unexpected(head_count.error());
    auto head_count_kv = get_int("qwen35.attention.head_count_kv");
    if (!head_count_kv) return std::unexpected(head_count_kv.error());
    auto key_length = get_int("qwen35.attention.key_length");
    if (!key_length) return std::unexpected(key_length.error());
    auto context_length = get_int("qwen35.context_length");
    if (!context_length) return std::unexpected(context_length.error());
    auto full_attention_interval = get_int("qwen35.full_attention_interval");
    if (!full_attention_interval) return std::unexpected(full_attention_interval.error());
    auto nextn_predict_layers = get_int("qwen35.nextn_predict_layers");
    if (!nextn_predict_layers) return std::unexpected(nextn_predict_layers.error());

    cfg.n_layers_ = *block_count - *nextn_predict_layers;
    if (cfg.n_layers_ <= 0) return std::unexpected(ErrorCode::ModelConfigInvalid);
    cfg.n_q_heads_ = *head_count;
    cfg.n_kv_heads_ = *head_count_kv;
    cfg.head_dim_ = *key_length;
    cfg.d_model_ = *embedding_length;
    cfg.d_ff_ = *feed_forward_length;
    cfg.max_seq_len_ = *context_length;
    cfg.full_attention_interval_ = *full_attention_interval;
    cfg.nextn_predict_layers_ = *nextn_predict_layers;

    if (auto ssm_conv = get_int("qwen35.ssm.conv_kernel"))
        cfg.ssm_conv_kernel_ = *ssm_conv;
    else
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    if (auto ssm_state = get_int("qwen35.ssm.state_size"))
        cfg.ssm_state_size_ = *ssm_state;
    else
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    if (auto ssm_group = get_int("qwen35.ssm.group_count"))
        cfg.ssm_group_count_ = *ssm_group;
    else
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    if (auto ssm_rank = get_int("qwen35.ssm.time_step_rank"))
        cfg.ssm_time_step_rank_ = *ssm_rank;
    else
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    if (auto ssm_inner = get_int("qwen35.ssm.inner_size"))
        cfg.ssm_inner_size_ = *ssm_inner;
    else
        return std::unexpected(ErrorCode::ModelConfigInvalid);

    if (auto freq = metadata_f32(reader, "qwen35.rope.freq_base"))
        cfg.rope_theta_ = *freq;
    else
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    if (auto rope_dim = get_int("qwen35.rope.dimension_count"))
        cfg.rotary_dim_ = *rope_dim;
    else
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    if (auto eps = metadata_f32(reader, "qwen35.attention.layer_norm_rms_epsilon")) {
        cfg.rms_norm_eps_ = *eps;
    } else {
        return std::unexpected(ErrorCode::ModelConfigInvalid);
    }

    // Decoder layer sequence: main layers first, then MTP predictor layers.
    cfg.layer_types_.reserve(static_cast<std::size_t>(*block_count));
    for (int i = 0; i < cfg.n_layers_; ++i) {
        if (*full_attention_interval > 0 &&
            i % *full_attention_interval == *full_attention_interval - 1) {
            cfg.layer_types_.push_back(LayerType::FullAttention);
        } else {
            cfg.layer_types_.push_back(LayerType::GatedDeltaNet);
        }
    }
    for (int i = 0; i < *nextn_predict_layers; ++i) {
        cfg.layer_types_.push_back(LayerType::MtpPredictor);
    }

    // Cross-check vocab with token_embd.weight and embedded tokenizer tokens.
    auto token_embd = reader.find_tensor("token_embd.weight");
    if (!token_embd.has_value() || token_embd->dims.empty()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }
    cfg.vocab_size_ = static_cast<int>(token_embd->dims.back());

    auto tokens = reader.metadata("tokenizer.ggml.tokens");
    if (tokens.has_value()) {
        const auto* arr = tokens->as_array();
        if (arr != nullptr && static_cast<int>(arr->size()) != cfg.vocab_size_) {
            return std::unexpected(ErrorCode::ModelConfigInvalid);
        }
    }

    auto validation = cfg.validate();
    if (!validation) return std::unexpected(validation.error());
    return cfg;
}

}  // namespace ccinfer
