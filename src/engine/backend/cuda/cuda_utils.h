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
        case cudaSuccess:
            return ErrorCode::Ok;
        case cudaErrorMemoryAllocation:
            return ErrorCode::CudaOutOfMemory;
        case cudaErrorInvalidValue:
            return ErrorCode::CudaInvalidValue;
        case cudaErrorInvalidDevice:
            return ErrorCode::CudaInvalidValue;
        case cudaErrorInvalidDevicePointer:
            return ErrorCode::CudaInvalidValue;
        case cudaErrorInvalidConfiguration:
            return ErrorCode::CudaInvalidValue;
        case cudaErrorLaunchFailure:
            return ErrorCode::CudaLaunchFailed;
        case cudaErrorLaunchOutOfResources:
            return ErrorCode::CudaLaunchFailed;
        case cudaErrorIllegalAddress:
            return ErrorCode::CudaLaunchFailed;
        default:
            return ErrorCode::CudaRuntimeError;
    }
}

inline Result<void> cuda_check(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return std::unexpected(map_cuda_error(err));
    }
    return {};
}

inline ErrorCode map_cublas_error(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_ALLOC_FAILED:
            return ErrorCode::CudaOutOfMemory;
        case CUBLAS_STATUS_INVALID_VALUE:
            return ErrorCode::CudaInvalidValue;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return ErrorCode::CudaLaunchFailed;
        default:
            return ErrorCode::CublasError;
    }
}

inline Result<void> cublas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %d\n", static_cast<int>(status));
        return std::unexpected(map_cublas_error(status));
    }
    return {};
}

// Check and clear the last CUDA error.  Call after every kernel launch.
inline Result<void> cuda_check_last_error() { return cuda_check(cudaGetLastError()); }

}  // namespace engine
}  // namespace ccinfer
