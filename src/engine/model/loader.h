#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtype.h"
#include "common/result.h"
#include "engine/backend/cuda/cuda_utils.h"
#include "engine/core/device_buffer.h"

namespace ccinfer {
namespace engine {

struct TensorInfo {
    DType dtype_ = DType::kBFloat16;
    std::vector<int64_t> shape_;
    uint64_t offset_ = 0;
    uint64_t size_bytes_ = 0;
};

template <typename T>
struct SafetensorDType;

template <>
struct SafetensorDType<__nv_bfloat16> {
    static constexpr DType value = DType::kBFloat16;
};

template <>
struct SafetensorDType<float> {
    static constexpr DType value = DType::kFloat32;
};

class WeightLoader {
public:
    static constexpr size_t kHeaderSize = 8;

    static Result<WeightLoader> create(const std::string& path);

    ~WeightLoader();

    WeightLoader(const WeightLoader&) = delete;
    WeightLoader& operator=(const WeightLoader&) = delete;

    WeightLoader(WeightLoader&& other) noexcept;
    WeightLoader& operator=(WeightLoader&& other) noexcept;

    bool has(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }

    size_t tensor_count() const noexcept { return tensors_.size(); }

    Result<TensorInfo> info(const std::string& name) const;

    template <typename T>
    Result<DeviceBuffer<T>> load(const std::string& name,
                                 const std::vector<int64_t>& expected_shape) const;

private:
    explicit WeightLoader(std::string path);

    Result<void> parse();
    void cleanup() noexcept;

    std::string path_;
    int fd_ = -1;
    uint8_t* data_ = nullptr;
    size_t size_ = 0;
    uint64_t header_len_ = 0;

    std::unordered_map<std::string, TensorInfo> tensors_;
};

// Template implementation.
template <typename T>
Result<DeviceBuffer<T>> WeightLoader::load(
    const std::string& name, const std::vector<int64_t>& expected_shape) const {
    if (data_ == nullptr || size_ == 0) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    const TensorInfo& info = it->second;

    if (info.shape_ != expected_shape) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    if (info.dtype_ != SafetensorDType<T>::value) {
        return std::unexpected(ErrorCode::ModelUnsupportedDType);
    }

    int64_t numel = 1;
    for (int64_t s : info.shape_) {
        if (s < 0) return std::unexpected(ErrorCode::ModelShapeMismatch);
        if (s == 0) return DeviceBuffer<T>{};
        if (numel > std::numeric_limits<int64_t>::max() / s) {
            return std::unexpected(ErrorCode::ModelShapeMismatch);
        }
        numel *= s;
    }

    const uint64_t expected_bytes = static_cast<uint64_t>(numel) * sizeof(T);
    if (info.size_bytes_ != expected_bytes) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    const uint64_t data_start = kHeaderSize + header_len_;
    if (data_start + info.offset_ > size_ ||
        info.size_bytes_ > size_ - data_start - info.offset_) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    const uint8_t* src = data_ + data_start + info.offset_;
    DeviceBuffer<T> buf(static_cast<size_t>(numel));

    auto r = cuda_check(cudaMemcpy(buf.get(), src, static_cast<size_t>(info.size_bytes_),
                                   cudaMemcpyHostToDevice));
    if (!r) return std::unexpected(r.error());

    return buf;
}

}  // namespace engine
}  // namespace ccinfer
