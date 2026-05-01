#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "common/result.h"

namespace ccinfer {
namespace engine {

class DeviceBackend;
class Model;
struct ModelConfig;
struct ModelWeights;

class ModelRunner {
public:
    ModelRunner();
    ~ModelRunner();

    Result<void> load_model(const std::string& model_path);

    // Placeholder — wired in Phase 3+.
    Result<void> forward(){ return {}; }

    DeviceBackend* backend() const noexcept { return backend_.get(); }
    Model* model() const noexcept { return model_.get(); }
    cudaStream_t stream() const noexcept { return stream_; }

private:
    std::unique_ptr<DeviceBackend> backend_;
    std::unique_ptr<Model> model_;
    cudaStream_t stream_{};
};

}  // namespace engine
}  // namespace ccinfer
