#pragma once

#include <memory>
#include <string>

#include "base/error.h"
#include "config/model_config.h"
#include "model/weight_source.h"

namespace ccinfer {

// Format-neutral checkpoint access. Implementations expose a canonical
// ModelConfig and a WeightSource; they do not allocate device memory.
class Checkpoint {
public:
    virtual ~Checkpoint() = default;

    virtual Result<ModelConfig> load_config() = 0;
    virtual WeightSource& weights() = 0;

    static Result<std::unique_ptr<Checkpoint>> open(const std::string& model_path);
};

}  // namespace ccinfer
