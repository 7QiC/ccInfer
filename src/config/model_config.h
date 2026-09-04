#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "base/error.h"

namespace ccinfer {

enum class ModelArch : uint8_t { Qwen3, Qwen3_5, Unknown };

enum class LayerType : uint8_t { FullAttention, GatedDeltaNet, MtpPredictor };

// Canonical model architecture description, independent of any checkpoint
// format. Loaders (HF config.json, GGUF metadata, ...) are responsible for
// translating external schemas into this struct.
struct ModelConfig {
    ModelArch arch_ = ModelArch::Unknown;

    int n_layers_ = 0;
    int n_q_heads_ = 0;
    int n_kv_heads_ = 0;
    int d_model_ = 0;
    int head_dim_ = 0;
    int rotary_dim_ = 0;
    int d_ff_ = 0;
    int vocab_size_ = 0;

    // Checkpoint/model-declared maximum context capability. This is not a
    // ccInfer runtime request limit (see EngineConfig::default_max_context_len).
    int max_seq_len_ = 0;

    float rope_theta_ = 10000.0f;
    float rms_norm_eps_ = 1e-6f;

    // Qwen3.5 hybrid decoder / MTP fields. Qwen3 leaves these at defaults.
    int full_attention_interval_ = 0;
    int nextn_predict_layers_ = 0;
    int ssm_conv_kernel_ = 0;
    int ssm_state_size_ = 0;
    int ssm_group_count_ = 0;
    int ssm_time_step_rank_ = 0;
    int ssm_inner_size_ = 0;

    // Full decoder layer sequence, including any MTP predictor layers at the
    // end. n_layers_ describes the main model layers (excluding MTP).
    std::vector<LayerType> layer_types_;

    const char* arch_name() const noexcept {
        switch (arch_) {
            case ModelArch::Qwen3:
                return "qwen3";
            case ModelArch::Qwen3_5:
                return "qwen35";
            default:
                return "unknown";
        }
    }

    // Validates canonical model invariants only. It does not validate external
    // schema presence/type; those are the config loader's responsibility.
    Result<void> validate() const;
};

}  // namespace ccinfer
