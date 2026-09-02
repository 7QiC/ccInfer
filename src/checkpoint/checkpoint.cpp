#include "checkpoint/checkpoint.h"

#include <filesystem>

#include "checkpoint/gguf/checkpoint.h"
#include "checkpoint/huggingface/checkpoint.h"

namespace ccinfer {

Result<std::unique_ptr<Checkpoint>> Checkpoint::open(const std::string& model_path) {
    namespace fs = std::filesystem;
    const fs::path path(model_path);

    std::error_code ec;
    if (fs::is_directory(path, ec)) {
        if (!fs::exists(path / "config.json") || !fs::exists(path / "model.safetensors")) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }
        return HuggingFaceCheckpoint::open(model_path);
    }

    if (fs::is_regular_file(path, ec) && path.extension() == ".gguf") {
        return GGUFCheckpoint::open(model_path);
    }

    return std::unexpected(ErrorCode::ModelLoadFailed);
}

}  // namespace ccinfer
