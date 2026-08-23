#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "backend/backend.h"
#include "base/result.h"
#include "core/tensor.h"

namespace ccinfer {

struct TensorInfo {
    ccop::DType dtype_ = ccop::DType::kBFloat16;
    std::vector<int64_t> shape_;
    uint64_t offset_ = 0;
    uint64_t size_bytes_ = 0;
};

template <typename T>
struct SafetensorDType;

template <>
struct SafetensorDType<__nv_bfloat16> {
    static constexpr ccop::DType value = ccop::DType::kBFloat16;
};

template <>
struct SafetensorDType<float> {
    static constexpr ccop::DType value = ccop::DType::kFloat32;
};

template <>
struct SafetensorDType<__half> {
    static constexpr ccop::DType value = ccop::DType::kFloat16;
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

    bool has(const std::string& name) const { return tensors_.find(name) != tensors_.end(); }

    size_t tensor_count() const noexcept { return tensors_.size(); }

    Result<TensorInfo> info(const std::string& name) const;

    template <typename T>
    Result<std::shared_ptr<Buffer>> load(Backend& backend, const std::string& name,
                                         const std::vector<int64_t>& expected_shape) const;

    template <typename T>
    Result<Tensor> load_tensor(Backend& backend, const std::string& name,
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

template <typename T>
Result<std::shared_ptr<Buffer>> WeightLoader::load(
    Backend& backend, const std::string& name, const std::vector<int64_t>& expected_shape) const {
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
        if (s <= 0) return std::unexpected(ErrorCode::ModelShapeMismatch);
        if (numel > std::numeric_limits<int64_t>::max() / s) {
            return std::unexpected(ErrorCode::ModelShapeMismatch);
        }
        numel *= s;
    }

    if (static_cast<uint64_t>(numel) > std::numeric_limits<uint64_t>::max() / sizeof(T)) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }
    const uint64_t expected_bytes = static_cast<uint64_t>(numel) * sizeof(T);
    if (info.size_bytes_ != expected_bytes) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }

    const uint64_t data_start = kHeaderSize + header_len_;
    if (data_start > static_cast<uint64_t>(size_)) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }
    if (info.offset_ > static_cast<uint64_t>(size_) - data_start) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }
    if (info.size_bytes_ > static_cast<uint64_t>(size_) - data_start - info.offset_) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    if (expected_bytes > std::numeric_limits<std::size_t>::max()) {
        return std::unexpected(ErrorCode::ModelShapeMismatch);
    }
    const auto copy_bytes = static_cast<std::size_t>(expected_bytes);

    const uint8_t* src = data_ + data_start + info.offset_;
    auto buf_r = backend.allocate_buffer(copy_bytes);
    if (!buf_r) return std::unexpected(buf_r.error());
    auto buf = std::move(*buf_r);

    auto r = backend.memcpy_h2d(buf->data(), src, copy_bytes);
    if (!r) return std::unexpected(r.error());

    auto sync_r = backend.synchronize();
    if (!sync_r) return std::unexpected(sync_r.error());

    return std::move(buf);
}

template <typename T>
Result<Tensor> WeightLoader::load_tensor(Backend& backend, const std::string& name,
                                         const std::vector<int64_t>& expected_shape) const {
    auto buffer = load<T>(backend, name, expected_shape);
    if (!buffer) return std::unexpected(buffer.error());
    auto tensor_buffer = std::move(*buffer);
    void* data = tensor_buffer->data();
    return Tensor::from_buffer(std::move(tensor_buffer), data, SafetensorDType<T>::value,
                               expected_shape);
}

}  // namespace ccinfer
