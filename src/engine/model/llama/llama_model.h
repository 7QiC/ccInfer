#pragma once

#include "engine/model/rope/rope_cache.h"
#include "engine/model/config.h"
#include "engine/model/model.h"
#include "engine/model/weights.h"

namespace ccinfer {
namespace engine {

class LlamaModel final : public Model {
public:
    LlamaModel(ModelConfig config, ModelWeights weights);

    Result<void> forward(const ForwardInput& input, ForwardOutput& output, DeviceBackend& backend,
                         void* stream) override;

    const ModelConfig& config() const override { return config_; }
    const char* architecture() const override { return "llama"; }

private:
    ModelConfig config_;
    ModelWeights weights_;
    RopeCache rope_cache_;
};

}  // namespace engine
}  // namespace ccinfer
