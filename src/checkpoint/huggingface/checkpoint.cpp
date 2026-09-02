#include "checkpoint/huggingface/checkpoint.h"

#include <utility>

#include "checkpoint/huggingface/config_loader.h"
#include "checkpoint/huggingface/reader.h"
#include "checkpoint/huggingface/weight_source.h"

namespace ccinfer {

HuggingFaceCheckpoint::HuggingFaceCheckpoint(
    std::string model_path, std::shared_ptr<Reader> reader,
    std::unique_ptr<SafetensorsWeightSource> weights)
    : model_path_(std::move(model_path)),
      reader_(std::move(reader)),
      weights_(std::move(weights)) {}

Result<std::unique_ptr<HuggingFaceCheckpoint>> HuggingFaceCheckpoint::open(
    const std::string& model_path) {
    auto reader_r = Reader::create(model_path + "/model.safetensors");
    if (!reader_r) return std::unexpected(reader_r.error());
    auto reader = std::shared_ptr<Reader>(std::move(*reader_r));

    auto weights = std::make_unique<SafetensorsWeightSource>(reader);
    return std::unique_ptr<HuggingFaceCheckpoint>(
        new HuggingFaceCheckpoint(model_path, std::move(reader), std::move(weights)));
}

Result<ModelConfig> HuggingFaceCheckpoint::load_config() {
    return HfConfigLoader::load(model_path_ + "/config.json");
}

WeightSource& HuggingFaceCheckpoint::weights() { return *weights_; }

}  // namespace ccinfer
