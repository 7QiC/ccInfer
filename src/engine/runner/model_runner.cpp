#include "engine/runner/model_runner.h"

#include "engine/backend/backend.h"
#include "engine/backend/cuda/cuda_utils.h"
#include "engine/model/model.h"

namespace ccinfer {
namespace engine {

ModelRunner::ModelRunner() {
    backend_ = DeviceBackend::create();

    if (auto r = cuda_check(cudaStreamCreate(&stream_)); !r) {
        std::abort();
    }
}

ModelRunner::~ModelRunner() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

Result<void> ModelRunner::load_model(const std::string& /*model_path*/) {
    // Placeholder — Phase 3 integrates WeightLoader + ModelRegistry.
    return {};
}

}  // namespace engine
}  // namespace ccinfer
