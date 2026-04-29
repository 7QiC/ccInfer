#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ccinfer {
namespace engine {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t n) : size_(n) { cudaMalloc(&ptr_, n * sizeof(T)); }

    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    operator T*() const noexcept { return ptr_; }
    size_t size() const noexcept { return size_; }
    size_t bytes() const noexcept { return size_ * sizeof(T); }
    bool empty() const noexcept { return size_ == 0; }

private:
    T* ptr_ = nullptr;
    size_t size_ = 0;
};

}  // namespace engine
}  // namespace ccinfer
