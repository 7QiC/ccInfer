#pragma once

#include <cstdint>

#include "common/result.h"

namespace ccinfer {
namespace engine {

struct ModelConfig;
class DeviceBackend;

struct ForwardInput {
    const void* input_embeds = nullptr;  // [T, D]
    int num_tokens = 0;
    const int32_t* positions = nullptr;  // optional; nullptr means generate 0..T-1
};

struct ForwardOutput {
    void* logits = nullptr;  // [T, vocab]
};

class Model {
public:
    virtual ~Model() = default;

    virtual Result<void> forward(const ForwardInput& input, ForwardOutput& output,
                                 DeviceBackend& backend, void* stream) = 0;

    virtual const ModelConfig& config() const = 0;
    virtual const char* architecture() const = 0;
};

}  // namespace engine
}  // namespace ccinfer
