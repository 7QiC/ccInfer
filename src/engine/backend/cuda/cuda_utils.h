#pragma once

#include <cstdio>
#include <cstdlib>

#include <source_location>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace ccinfer {
namespace engine {

inline void cuda_check_impl(cudaError_t err, const std::source_location& loc) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[%s:%d] CUDA error: %s\n", loc.file_name(), (int)loc.line(),
                cudaGetErrorString(err));
        abort();
    }
}

inline void cublas_check_impl(cublasStatus_t status, const std::source_location& loc) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[%s:%d] cuBLAS error: %d\n", loc.file_name(), (int)loc.line(),
                (int)status);
        abort();
    }
}

}  // namespace engine
}  // namespace ccinfer

#define CCINFER_CUDA_CHECK(call) \
    ccinfer::engine::cuda_check_impl((call), std::source_location::current())

#define CCINFER_CUBLAS_CHECK(call) \
    ccinfer::engine::cublas_check_impl((call), std::source_location::current())
