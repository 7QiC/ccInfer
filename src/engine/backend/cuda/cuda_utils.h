#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>

#include "common/error_code.h"
#include "common/result.h"

namespace ccinfer {
namespace engine {

inline ErrorCode map_cuda_error(cudaError_t err) {
    switch (err) {
        case cudaErrorMemoryAllocation:
            return ErrorCode::CudaOutOfMemory;
        case cudaErrorInvalidValue:
            return ErrorCode::CudaInvalidValue;
        default:
            return ErrorCode::CudaLaunchFailed;
    }
}

inline Result<void> cuda_check(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return std::unexpected(map_cuda_error(err));
    }
    return {};
}

inline Result<void> cublas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(status));
        return std::unexpected(ErrorCode::CudaLaunchFailed);
    }
    return {};
}

}  // namespace engine
}  // namespace ccinfer
