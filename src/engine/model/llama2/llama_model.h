#pragma once

#include <memory>

#include "engine/model/llama2/llama2_weights.h"
#include "engine/model/model.h"
#include "engine/model/rope/rope_cache.h"

namespace ccinfer {
namespace engine {

class LlamaModel final : public Model {
public:
    static Result<std::unique_ptr<Model>> create(const ModelConfig& config,
                                                  const WeightLoader& loader,
                                                  DefaultBackend& backend);

    LlamaModel(ModelConfig config, LlamaWeights weights, RopeCache rope_cache);

    Result<void> forward(const ForwardInput& input, ForwardOutput& output,
                         DefaultBackend& backend) override;

    const ModelConfig& config() const override { return config_; }
    const char* architecture() const override { return "llama"; }

private:
    ModelConfig config_;
    LlamaWeights weights_;
    RopeCache rope_cache_;
};

}  // namespace engine
}  // namespace ccinfer
