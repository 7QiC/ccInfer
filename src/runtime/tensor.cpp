#include "runtime/tensor.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <utility>

#include "backend/backend.h"
#include "base/error.h"

namespace ccinfer {

namespace {

bool data_in_buffer(const Buffer* buffer, const void* data, std::size_t bytes) noexcept {
    const auto base = reinterpret_cast<std::uintptr_t>(buffer->data());
    const auto ptr = reinterpret_cast<std::uintptr_t>(data);
    return ptr >= base && bytes <= buffer->bytes() &&
           (ptr - base) <= buffer->bytes() - bytes;
}

Result<std::size_t> checked_numel(std::span<const std::int64_t> shape) {
    if (shape.empty() || shape.size() > static_cast<std::size_t>(ccop::kTensorMaxRank)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
    std::size_t numel = 1;
    for (const std::int64_t dim : shape) {
        if (dim <= 0 || static_cast<std::uint64_t>(dim) > kMax / numel) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }
        numel *= static_cast<std::size_t>(dim);
    }
    return numel;
}

Result<std::size_t> checked_nbytes_dense(ccop::DType dtype, std::span<const std::int64_t> shape) {
    const std::size_t elem_size = ccop::dtype_size(dtype);
    if (elem_size == 0) return std::unexpected(ErrorCode::Unsupported);
    auto numel = checked_numel(shape);
    if (!numel) return std::unexpected(numel.error());
    if (*numel > std::numeric_limits<std::size_t>::max() / elem_size) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    return *numel * elem_size;
}

Result<std::size_t> checked_nbytes_quantized(const ccop::QType& qtype,
                                             std::span<const std::int64_t> shape) {
    if (shape.empty() || shape.size() > static_cast<std::size_t>(ccop::kTensorMaxRank)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (qtype.quant_type != ccop::QuantType::kQ8_0) {
        return std::unexpected(ErrorCode::Unsupported);
    }
    auto bytes = ccop::q8_0_storage_bytes(shape);
    if (!bytes.has_value() || *bytes < 0) return std::unexpected(ErrorCode::InvalidArgument);
    return static_cast<std::size_t>(*bytes);
}

}  // namespace

Tensor::Tensor(std::shared_ptr<Buffer> buffer, ccop::DType dtype,
               std::initializer_list<std::int64_t> shape)
    : ccop::Tensor(buffer->data(), dtype, buffer->device(), shape), buffer_(std::move(buffer)) {
    assert(nbytes() <= buffer_->bytes());
}

Tensor::Tensor(std::shared_ptr<Buffer> buffer, const ccop::QType& qtype,
               std::initializer_list<std::int64_t> shape)
    : ccop::Tensor(buffer->data(), qtype, buffer->device(), shape), buffer_(std::move(buffer)) {
    assert(nbytes() <= buffer_->bytes());
}

Result<Tensor> Tensor::empty(Backend& backend, ccop::DType dtype,
                             std::initializer_list<std::int64_t> shape) {
    return empty(backend, dtype, std::span<const std::int64_t>(shape.begin(), shape.size()));
}

Result<Tensor> Tensor::empty(Backend& backend, ccop::DType dtype,
                             std::span<const std::int64_t> shape) {
    auto bytes = checked_nbytes_dense(dtype, shape);
    if (!bytes) return std::unexpected(bytes.error());
    auto buffer_r = backend.allocate_buffer(*bytes);
    if (!buffer_r) return std::unexpected(buffer_r.error());
    auto buffer = std::move(*buffer_r);
    return from_buffer(buffer, buffer->data(), dtype, shape);
}

Result<Tensor> Tensor::empty(Backend& backend, const ccop::QType& qtype,
                             std::initializer_list<std::int64_t> shape) {
    auto bytes = checked_nbytes_quantized(
        qtype, std::span<const std::int64_t>(shape.begin(), shape.size()));
    if (!bytes) return std::unexpected(bytes.error());
    auto buffer_r = backend.allocate_buffer(*bytes);
    if (!buffer_r) return std::unexpected(buffer_r.error());
    auto buffer = std::move(*buffer_r);
    return from_buffer(buffer, buffer->data(), qtype, shape);
}

Result<Tensor> Tensor::from_host(Backend& backend, const void* src, ccop::DType dtype,
                                 std::initializer_list<std::int64_t> shape) {
    return from_host(backend, src, dtype,
                     std::span<const std::int64_t>(shape.begin(), shape.size()));
}

Result<Tensor> Tensor::from_host(Backend& backend, const void* src, ccop::DType dtype,
                                 std::span<const std::int64_t> shape) {
    if (src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
    auto tensor = empty(backend, dtype, shape);
    if (!tensor) return std::unexpected(tensor.error());
    auto r = backend.memcpy_h2d(tensor->data(), src, tensor->nbytes());
    if (!r) return std::unexpected(r.error());
    return tensor;
}

Result<Tensor> Tensor::from_host(Backend& backend, const void* src, const ccop::QType& qtype,
                                 std::initializer_list<std::int64_t> shape) {
    if (src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
    auto tensor = empty(backend, qtype, shape);
    if (!tensor) return std::unexpected(tensor.error());
    auto r = backend.memcpy_h2d(tensor->data(), src, tensor->nbytes());
    if (!r) return std::unexpected(r.error());
    return tensor;
}

Tensor Tensor::from_buffer(std::shared_ptr<Buffer> buffer, void* data, ccop::DType dtype,
                           std::initializer_list<std::int64_t> shape) {
    Tensor tensor;
    static_cast<ccop::Tensor&>(tensor) = ccop::Tensor(data, dtype, buffer->device(), shape);
    tensor.buffer_ = std::move(buffer);
    assert(data_in_buffer(tensor.buffer_.get(), data, tensor.nbytes()));
    return tensor;
}

Tensor Tensor::from_buffer(std::shared_ptr<Buffer> buffer, void* data, ccop::DType dtype,
                           std::span<const std::int64_t> shape) {
    assert(shape.size() <= static_cast<std::size_t>(ccop::kTensorMaxRank));
    std::array<std::int64_t, ccop::kTensorMaxRank> shape_arr{};
    std::array<std::int64_t, ccop::kTensorMaxRank> stride{};
    std::copy(shape.begin(), shape.end(), shape_arr.begin());
    std::int64_t st = 1;
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        stride[static_cast<std::size_t>(d)] = st;
        st *= shape_arr[static_cast<std::size_t>(d)];
    }
    Tensor tensor;
    static_cast<ccop::Tensor&>(tensor) = ccop::Tensor(data, dtype, buffer->device(), shape_arr,
                                                      stride, static_cast<int>(shape.size()));
    tensor.buffer_ = std::move(buffer);
    assert(data_in_buffer(tensor.buffer_.get(), data, tensor.nbytes()));
    return tensor;
}

Tensor Tensor::from_buffer(std::shared_ptr<Buffer> buffer, void* data, const ccop::QType& qtype,
                           std::initializer_list<std::int64_t> shape) {
    Tensor tensor;
    static_cast<ccop::Tensor&>(tensor) = ccop::Tensor(data, qtype, buffer->device(), shape);
    tensor.buffer_ = std::move(buffer);
    assert(data_in_buffer(tensor.buffer_.get(), data, tensor.nbytes()));
    return tensor;
}

Tensor Tensor::view(std::initializer_list<std::int64_t> shape) {
    assert(buffer_ && "Tensor has no owner");
    if (!is_dense()) {
        assert(false && "view requires a dense tensor");
        return Tensor{};
    }
    assert(is_contiguous() && "view requires a contiguous tensor");
    const ccop::Tensor view(data(), dtype(), device(), shape);
    assert(view.numel() == numel() && "view shape must preserve numel");
    assert(view.nbytes() <= buffer_->bytes());
    return with_view(view);
}

Tensor Tensor::flat() { return view({numel()}); }

Tensor Tensor::slice(int dim, std::int64_t start, std::int64_t end) {
    return with_view(ccop::Tensor::slice(dim, start, end));
}

Tensor Tensor::select(int dim, std::int64_t index) {
    return with_view(ccop::Tensor::select(dim, index));
}

}  // namespace ccinfer
