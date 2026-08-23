#pragma once

#include <memory>

#include "model/config.h"
#include "model/model.h"
#include "model/qwen3/qwen3_weights.h"
#include "model/rope/rope_cache.h"

namespace ccinfer {

class Qwen3Model final : public Model {
public:
    static Result<std::unique_ptr<Model>> create(const ModelConfig& config,
                                                 const WeightLoader& loader,
                                                 Backend& backend);

    Qwen3Model(ModelConfig config, Qwen3Weights weights, RopeCache rope_cache);

    Result<void> forward(const ForwardInput& input, ForwardOutput& output,
                         Backend& backend) override;

    const ModelConfig& config() const override { return config_; }

private:
    ModelConfig config_;
    Qwen3Weights weights_;
    RopeCache rope_cache_;
};

}  // namespace ccinfer
