#pragma once

#include <memory>
#include <string>

#include "base/error.h"
#include "checkpoint/gguf/reader.h"
#include "checkpoint/gguf/weight_source.h"
#include "checkpoint/checkpoint.h"

namespace ccinfer {

class GGUFCheckpoint final : public Checkpoint {
public:
    static Result<std::unique_ptr<GGUFCheckpoint>> open(const std::string& path);

    Result<ModelConfig> load_config() override;
    WeightSource& weights() override;

private:
    GGUFCheckpoint(std::shared_ptr<GGUFReader> reader,
                   std::unique_ptr<GGUFWeightSource> weights);

    std::shared_ptr<GGUFReader> reader_;
    std::unique_ptr<GGUFWeightSource> weights_;
};

}  // namespace ccinfer
