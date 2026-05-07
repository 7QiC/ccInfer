#pragma once

#include <cstddef>

#include "engine/backend/device_buffer.h"

namespace ccinfer {
namespace engine {

class CudaBuffer final : public DeviceBuffer {
public:
    explicit CudaBuffer(size_t bytes);
    ~CudaBuffer() override;

    void* data() override { return ptr_; }
    const void* data() const override { return ptr_; }
    size_t bytes() const override { return bytes_; }

private:
    void* ptr_ = nullptr;
    size_t bytes_ = 0;
};

}  // namespace engine
}  // namespace ccinfer
