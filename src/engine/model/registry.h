#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "common/result.h"
#include "engine/model/config.h"
#include "engine/model/model.h"
#include "engine/model/weights.h"

namespace ccinfer {
namespace engine {

using ModelCreator = Result<std::unique_ptr<Model>> (*)(ModelConfig config, ModelWeights weights);

class ModelRegistry {
public:
    static ModelRegistry& instance();

    void register_model(std::string arch, ModelCreator creator);

    Result<std::unique_ptr<Model>> create(const ModelConfig& config, ModelWeights weights) const;

private:
    ModelRegistry() = default;

    std::unordered_map<std::string, ModelCreator> creators_;
};

void register_builtin_models();

}  // namespace engine
}  // namespace ccinfer
