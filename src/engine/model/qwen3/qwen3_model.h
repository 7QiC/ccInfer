#pragma once

#include <memory>

#include "engine/model/config.h"
#include "engine/model/model.h"
#include "engine/model/qwen3/qwen3_weights.h"
#include "engine/model/rope/rope_cache.h"

namespace ccinfer {
namespace engine {

class Qwen3Model final : public Model {
public:
    static Result<std::unique_ptr<Model>> create(const ModelConfig& config,
                                                  const WeightLoader& loader,
                                                  DeviceBackend& backend);

    Qwen3Model(ModelConfig config, Qwen3Weights weights, RopeCache rope_cache);

    Result<void> forward(const ForwardInput& input, ForwardOutput& output,
                         DeviceBackend& backend) override;

    const ModelConfig& config() const override { return config_; }
    const char* architecture() const override { return "qwen3"; }

private:
    ModelConfig config_;
    Qwen3Weights weights_;
    RopeCache rope_cache_;
};

}  // namespace engine
}  // namespace ccinfer
