#include "model/registry.h"

#include <utility>

#include "base/error_code.h"
#include "model/qwen3/qwen3_model.h"

namespace ccinfer {

ModelRegistry& ModelRegistry::instance() {
    static ModelRegistry registry;
    return registry;
}

void ModelRegistry::register_model(std::string arch, ModelCreator creator) {
    creators_[std::move(arch)] = creator;
}

Result<std::unique_ptr<Model>> ModelRegistry::create(const ModelConfig& config,
                                                     const WeightLoader& loader,
                                                     DefaultBackend& backend) const {
    auto it = creators_.find(config.arch_name());
    if (it == creators_.end()) {
        return std::unexpected(ErrorCode::ModelUnsupportedArch);
    }

    return it->second(config, loader, backend);
}

void register_builtin_models() {
    auto& r = ModelRegistry::instance();

    r.register_model("qwen3", &Qwen3Model::create);
}

}  // namespace ccinfer
