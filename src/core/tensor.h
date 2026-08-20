#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>

#include "backend/buffer.h"
#include "base/result.h"

namespace ccinfer {

class Backend;

// Framework-side Tensor: a ccop::Tensor view plus the Buffer that owns the
// underlying allocation. Copies share ownership (PyTorch-style value
// semantics); views produced by view/flat/slice/select keep the same owner.
class Tensor final : public ops::Tensor {
public:
    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    // Wraps an already-allocated Buffer. Never fails.
    Tensor(std::shared_ptr<Buffer> buffer, ops::DType dtype,
           std::initializer_list<std::int64_t> shape);

    // Uninitialized device tensor (torch.empty style).
    static Result<Tensor> empty(Backend& backend, ops::DType dtype,
                                std::initializer_list<std::int64_t> shape);

    // Allocates a device tensor and copies host data into it.
    static Result<Tensor> from_host(Backend& backend, const void* src, ops::DType dtype,
                                    std::initializer_list<std::int64_t> shape);

    // Wraps a sub-range of an existing Buffer at an explicit byte offset.
    static Tensor from_buffer(std::shared_ptr<Buffer> buffer, void* data, ops::DType dtype,
                              std::initializer_list<std::int64_t> shape);
    static Tensor from_buffer(std::shared_ptr<Buffer> buffer, void* data, ops::DType dtype,
                              std::span<const std::int64_t> shape);

    [[nodiscard]] const std::shared_ptr<Buffer>& buffer() const noexcept { return buffer_; }

    Tensor view(std::initializer_list<std::int64_t> shape) const;
    Tensor flat() const;
    Tensor slice(int dim, std::int64_t start, std::int64_t end) const;
    Tensor select(int dim, std::int64_t index) const;

private:
    Tensor with_view(const ops::Tensor& view) const {
        Tensor t = *this;
        static_cast<ops::Tensor&>(t) = view;
        return t;
    }

    std::shared_ptr<Buffer> buffer_;
};

}  // namespace ccinfer
