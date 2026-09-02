#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "base/error.h"
#include "config/model_config.h"
#include "model/model.h"
#include "model/weight_source.h"

namespace ccinfer {

using ModelCreator = Result<std::unique_ptr<Model>> (*)(const ModelConfig& config,
                                                        WeightSource& weights,
                                                        Backend& backend);

class ModelRegistry {
public:
    static ModelRegistry& instance();

    void register_model(std::string arch, ModelCreator creator);

    Result<std::unique_ptr<Model>> create(const ModelConfig& config, WeightSource& weights,
                                          Backend& backend) const;

private:
    ModelRegistry() = default;

    std::unordered_map<std::string, ModelCreator> creators_;
};

void register_builtin_models();

}  // namespace ccinfer
