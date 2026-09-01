#pragma once

#include <cstdint>
#include <string>

#include "base/error.h"

namespace ccinfer {

enum class ModelArch : uint8_t { Qwen3, Unknown };

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
    int d_ff_ = 0;
    int vocab_size_ = 0;

    // Checkpoint/model-declared maximum context capability. This is not a
    // ccInfer runtime request limit (see EngineConfig::default_max_context_len).
    int max_seq_len_ = 0;

    float rope_theta_ = 10000.0f;
    float rms_norm_eps_ = 1e-6f;

    const char* arch_name() const noexcept {
        switch (arch_) {
            case ModelArch::Qwen3:
                return "qwen3";
            default:
                return "unknown";
        }
    }

    // Validates canonical model invariants only. It does not validate external
    // schema presence/type; those are the config loader's responsibility.
    Result<void> validate() const;
};

}  // namespace ccinfer
