#include "checkpoint/gguf/checkpoint.h"

#include <utility>

#include "checkpoint/gguf/config_loader.h"

namespace ccinfer {

GGUFCheckpoint::GGUFCheckpoint(std::shared_ptr<GGUFReader> reader,
                               std::unique_ptr<GGUFWeightSource> weights)
    : reader_(std::move(reader)), weights_(std::move(weights)) {}

Result<std::unique_ptr<GGUFCheckpoint>> GGUFCheckpoint::open(const std::string& path) {
    auto reader_r = GGUFReader::create(path);
    if (!reader_r) return std::unexpected(reader_r.error());
    auto reader = std::shared_ptr<GGUFReader>(std::move(*reader_r));

    auto weights = std::make_unique<GGUFWeightSource>(reader);
    return std::unique_ptr<GGUFCheckpoint>(
        new GGUFCheckpoint(std::move(reader), std::move(weights)));
}

Result<ModelConfig> GGUFCheckpoint::load_config() {
    return GgufConfigLoader::load(*reader_);
}

WeightSource& GGUFCheckpoint::weights() { return *weights_; }

}  // namespace ccinfer
