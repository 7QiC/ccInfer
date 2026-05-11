#include "engine/backend/cuda/cuda_buffer.h"

#include <cuda_runtime.h>

#include "common/error_code.h"
#include "engine/backend/cuda/cuda_utils.h"

namespace ccinfer {
namespace engine {

Result<std::unique_ptr<CudaBuffer>> CudaBuffer::create(size_t bytes) {
    if (bytes == 0) return std::unexpected(ErrorCode::InvalidArgument);
    void* ptr = nullptr;
    auto r = cuda_check(cudaMalloc(&ptr, bytes));
    if (!r) return std::unexpected(r.error());
    return std::unique_ptr<CudaBuffer>(new CudaBuffer(bytes, ptr));
}

CudaBuffer::~CudaBuffer() {
    if (ptr_) cudaFree(ptr_);
}

}  // namespace engine
}  // namespace ccinfer
