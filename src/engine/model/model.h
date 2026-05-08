#pragma once

#include <cstdint>

#include "common/result.h"
#include "engine/common/backend_def.h"

namespace ccinfer {
namespace engine {

struct ModelConfig;

struct ForwardInput {
    const void* input_embeds_ = nullptr;  // [T, D]
    int num_tokens_ = 0;
    const int32_t* positions_ = nullptr;  // optional; nullptr means generate 0..T-1
};

struct ForwardOutput {
    void* logits_ = nullptr;  // [T, vocab]
};

class Model {
public:
    virtual ~Model() = default;

    virtual Result<void> forward(const ForwardInput& input, ForwardOutput& output,
                                 DefaultBackend& backend) = 0;

    virtual const ModelConfig& config() const = 0;
    virtual const char* architecture() const = 0;
};

}  // namespace engine
}  // namespace ccinfer
