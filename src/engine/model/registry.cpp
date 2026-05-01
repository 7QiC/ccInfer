#include "engine/model/registry.h"

namespace ccinfer {
namespace engine {

ModelRegistry& ModelRegistry::instance() {
    static ModelRegistry registry;
    return registry;
}

void ModelRegistry::register_model(std::string arch, ModelCreator creator) {
    creators_.emplace(std::move(arch), creator);
}

Result<std::unique_ptr<Model>> ModelRegistry::create(const ModelConfig& config,
                                                     ModelWeights weights) const {
    auto it = creators_.find(config.arch_name());
    if (it == creators_.end()) {
        return std::unexpected(ErrorCode::ModelUnsupportedArch);
    }

    return it->second(config, std::move(weights));
}

void register_builtin_models() {
    // Registered in llama_model.cpp via static initializer.
    // This function is a hook for explicit registration if needed.
}

}  // namespace engine
}  // namespace ccinfer
