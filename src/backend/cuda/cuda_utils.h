#pragma once

#include <cublas_v2.h>

#include <cuda_runtime.h>

#include "common/error_code.h"
#include "facade/log.h"

namespace ccinfer {

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
        ccLog::error("CUDA error: {}", cudaGetErrorString(err));
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
        ccLog::error("cuBLAS error: {}", static_cast<int>(status));
        return std::unexpected(map_cublas_error(status));
    }
    return {};
}

}  // namespace ccinfer
