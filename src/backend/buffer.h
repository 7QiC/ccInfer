#pragma once

#include <cstddef>

#include "ops/ops.h"

namespace ccinfer {

// Framework-side memory ownership object.
//
// Buffer is the ONLY object that owns device/host memory. ops::Tensor is a
// pure view that borrows Buffer::data(); a Tensor must not outlive the Buffer
// it was created from (std::string_view-style borrowing). The concrete
// allocation (e.g. cudaMalloc) lives in the selected backend implementation.
class Buffer {
public:
    virtual ~Buffer() = default;

    [[nodiscard]] virtual void* data() noexcept = 0;
    [[nodiscard]] virtual const void* data() const noexcept = 0;
    [[nodiscard]] virtual std::size_t bytes() const noexcept = 0;
    [[nodiscard]] virtual ops::Device device() const noexcept = 0;

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(Buffer&&) = delete;

protected:
    Buffer() = default;
};

}  // namespace ccinfer
