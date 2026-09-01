#pragma once

#include <string>

#include <nlohmann/json.hpp>

#include "base/error.h"
#include "config/model_config.h"

namespace ccinfer {

// Translates HuggingFace config.json into the canonical ModelConfig. This is
// the only place that understands the HF JSON schema.
class HfConfigLoader {
public:
    static Result<ModelConfig> load(const std::string& config_json_path);
    static Result<ModelConfig> load_from_json(const nlohmann::json& j);
};

}  // namespace ccinfer
