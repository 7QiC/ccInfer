#pragma once

#include <cstddef>

namespace ccinfer {

class DeviceBuffer {
public:
    virtual ~DeviceBuffer() = default;

    virtual void* data() = 0;
    virtual const void* data() const = 0;
    virtual size_t bytes() const = 0;

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

protected:
    DeviceBuffer() = default;
};

// Typed accessor helpers.
template <typename T>
T* buffer_data(DeviceBuffer& buf) {
    return static_cast<T*>(buf.data());
}
template <typename T>
const T* buffer_data(const DeviceBuffer& buf) {
    return static_cast<const T*>(buf.data());
}

}  // namespace ccinfer
