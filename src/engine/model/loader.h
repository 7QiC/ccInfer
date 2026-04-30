#pragma once

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtype.h"
#include "common/result.h"
#include "engine/core/device_buffer.h"
#include "engine/model/config.h"
#include "engine/model/weights.h"

namespace ccinfer {
namespace engine {

struct TensorInfo {
    DType dtype_ = DType::kBFloat16;
    std::vector<int64_t> shape_;
    uint64_t offset_ = 0;  // offset relative to safetensors data section
    uint64_t size_bytes_ = 0;
};

class WeightLoader {
public:
    static Result<WeightLoader> create(const std::string& path);

    ~WeightLoader();

    WeightLoader(const WeightLoader&) = delete;
    WeightLoader& operator=(const WeightLoader&) = delete;

    WeightLoader(WeightLoader&& other) noexcept;
    WeightLoader& operator=(WeightLoader&& other) noexcept;

    [[nodiscard]] bool has(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }

    [[nodiscard]] size_t tensor_count() const noexcept { return tensors_.size(); }

    Result<DeviceBuffer<__nv_bfloat16>> load(const std::string& name,
                                             const std::vector<int64_t>& expected_shape) const;

    Result<ModelWeights> load_all(const ModelConfig& config) const;

private:
    explicit WeightLoader(std::string path);

    Result<void> parse();
    void cleanup() noexcept;

private:
    std::string path_;
    int fd_ = -1;
    uint8_t* data_ = nullptr;
    size_t size_ = 0;
    uint64_t header_len_ = 0;

    std::unordered_map<std::string, TensorInfo> tensors_;
};

}  // namespace engine
}  // namespace ccinfer
