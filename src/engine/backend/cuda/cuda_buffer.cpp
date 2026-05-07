#include "engine/backend/cuda/cuda_buffer.h"

#include <cuda_runtime.h>

namespace ccinfer {
namespace engine {

CudaBuffer::CudaBuffer(size_t bytes) : bytes_(bytes) {
    cudaMalloc(&ptr_, bytes);
}

CudaBuffer::~CudaBuffer() {
    if (ptr_) cudaFree(ptr_);
}

}  // namespace engine
}  // namespace ccinfer
