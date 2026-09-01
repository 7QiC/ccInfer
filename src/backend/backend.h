#pragma once

#include <cstddef>
#include <memory>

#include "backend/buffer.h"
#include "base/error.h"
#include "facade/ops.h"

namespace ccinfer {

// Framework execution-entry: a single plain Backend class (PIMPL).
//
// The backend only owns execution resources (stream + cublas handle) and
// memory transfer/allocation. All operators live in ccop and are invoked
// through ccop::Tensor views.
class Backend final {
public:
    static Result<std::unique_ptr<Backend>> create(int device_id);

    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    Backend(Backend&&) noexcept;
    Backend& operator=(Backend&&) noexcept;

    Result<std::shared_ptr<Buffer>> allocate_buffer(std::size_t bytes);
    Result<void> memcpy_h2d(void* dst, const void* src, std::size_t count);
    Result<void> memcpy_d2h(void* dst, const void* src, std::size_t count);
    Result<void> memcpy_d2d(void* dst, const void* src, std::size_t count);
    Result<void> memset(void* dst, int value, std::size_t count);
    Result<void> synchronize();
    [[nodiscard]] ccop::ExecutionContext context() const noexcept;

private:
    struct Impl;
    explicit Backend(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace ccinfer
