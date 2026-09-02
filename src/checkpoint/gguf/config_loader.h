#pragma once

#include "base/error.h"
#include "checkpoint/gguf/reader.h"
#include "config/model_config.h"

namespace ccinfer {

class GgufConfigLoader {
public:
    static Result<ModelConfig> load(const GGUFReader& reader);
};

}  // namespace ccinfer
