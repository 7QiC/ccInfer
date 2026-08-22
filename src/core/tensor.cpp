#include "core/tensor.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <utility>

#include "backend/backend.h"
#include "base/error_code.h"

namespace ccinfer {

namespace {

Result<std::size_t> checked_nbytes(ops::DType dtype, std::initializer_list<std::int64_t> shape) {
    const std::size_t elem_size = ops::dtype_size(dtype);
    if (elem_size == 0) return std::unexpected(ErrorCode::Unsupported);
    if (shape.size() == 0 || shape.size() > static_cast<std::size_t>(ops::kTensorMaxRank)) {
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
    if (numel > kMax / elem_size) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    return numel * elem_size;
}

}  // namespace

Tensor::Tensor(std::shared_ptr<Buffer> buffer, ops::DType dtype,
               std::initializer_list<std::int64_t> shape)
    : ops::Tensor(buffer->data(), dtype, buffer->device(), shape), buffer_(std::move(buffer)) {
    assert(nbytes() <= buffer_->bytes());
}

Result<Tensor> Tensor::empty(Backend& backend, ops::DType dtype,
                             std::initializer_list<std::int64_t> shape) {
    auto bytes = checked_nbytes(dtype, shape);
    if (!bytes) return std::unexpected(bytes.error());
    auto buffer = backend.allocate_buffer(*bytes);
    if (!buffer) return std::unexpected(buffer.error());
    return Tensor(std::move(*buffer), dtype, shape);
}

Result<Tensor> Tensor::from_host(Backend& backend, const void* src, ops::DType dtype,
                                 std::initializer_list<std::int64_t> shape) {
    if (src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
    auto tensor = empty(backend, dtype, shape);
    if (!tensor) return std::unexpected(tensor.error());
    auto r = backend.memcpy_h2d(tensor->data(), src, tensor->nbytes());
    if (!r) return std::unexpected(r.error());
    return tensor;
}

Tensor Tensor::from_buffer(std::shared_ptr<Buffer> buffer, void* data, ops::DType dtype,
                           std::initializer_list<std::int64_t> shape) {
    Tensor tensor;
    static_cast<ops::Tensor&>(tensor) = ops::Tensor(data, dtype, buffer->device(), shape);
    tensor.buffer_ = std::move(buffer);
    assert(tensor.nbytes() <= tensor.buffer_->bytes());
    return tensor;
}

Tensor Tensor::from_buffer(std::shared_ptr<Buffer> buffer, void* data, ops::DType dtype,
                           std::span<const std::int64_t> shape) {
    assert(shape.size() <= static_cast<std::size_t>(ops::kTensorMaxRank));
    std::array<std::int64_t, ops::kTensorMaxRank> shape_arr{};
    std::array<std::int64_t, ops::kTensorMaxRank> stride{};
    std::copy(shape.begin(), shape.end(), shape_arr.begin());
    std::int64_t st = 1;
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        stride[static_cast<std::size_t>(d)] = st;
        st *= shape_arr[static_cast<std::size_t>(d)];
    }
    Tensor tensor;
    static_cast<ops::Tensor&>(tensor) = ops::Tensor(data, dtype, buffer->device(), shape_arr,
                                                    stride, static_cast<int>(shape.size()));
    tensor.buffer_ = std::move(buffer);
    assert(tensor.nbytes() <= tensor.buffer_->bytes());
    return tensor;
}

Tensor Tensor::view(std::initializer_list<std::int64_t> shape) const {
    assert(buffer_ && "Tensor has no owner");
    const ops::Tensor view(const_cast<void*>(data()), dtype(), device(), shape);
    assert(view.nbytes() <= buffer_->bytes());
    return with_view(view);
}

Tensor Tensor::flat() const { return view({numel()}); }

Tensor Tensor::slice(int dim, std::int64_t start, std::int64_t end) const {
    return with_view(ops::Tensor::slice(dim, start, end));
}

Tensor Tensor::select(int dim, std::int64_t index) const {
    return with_view(ops::Tensor::select(dim, index));
}

}  // namespace ccinfer
