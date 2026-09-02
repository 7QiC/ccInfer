#pragma once

#include <memory>
#include <string>

#include "base/error.h"
#include "checkpoint/checkpoint.h"
#include "checkpoint/huggingface/reader.h"
#include "checkpoint/huggingface/weight_source.h"

namespace ccinfer {

class HuggingFaceCheckpoint final : public Checkpoint {
public:
    static Result<std::unique_ptr<HuggingFaceCheckpoint>> open(const std::string& model_path);

    Result<ModelConfig> load_config() override;
    WeightSource& weights() override;

private:
    HuggingFaceCheckpoint(std::string model_path,
                          std::shared_ptr<Reader> reader,
                          std::unique_ptr<SafetensorsWeightSource> weights);

    std::string model_path_;
    std::shared_ptr<Reader> reader_;
    std::unique_ptr<SafetensorsWeightSource> weights_;
};

}  // namespace ccinfer
