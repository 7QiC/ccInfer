#pragma once

#include <cstddef>
#include <memory>

#include "base/result.h"
#include "backend/device_buffer.h"

namespace ccinfer {

class CudaBuffer final : public DeviceBuffer {
public:
    static Result<std::unique_ptr<CudaBuffer>> create(size_t bytes);

    ~CudaBuffer() override;

    void* data() override { return ptr_; }
    const void* data() const override { return ptr_; }
    size_t bytes() const override { return bytes_; }

private:
    explicit CudaBuffer(size_t bytes, void* ptr) : bytes_(bytes), ptr_(ptr) {}

    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};

}  // namespace ccinfer
