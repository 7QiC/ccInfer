#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <source_location>

inline void cuda_check_impl(cudaError_t err, const std::source_location& loc) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[%s:%d] CUDA error: %s\n",
                loc.file_name(), (int)loc.line(), cudaGetErrorString(err));
        abort();
    }
}

#define CUDA_CHECK(call) cuda_check_impl((call), std::source_location::current())

inline void cublas_check_impl(cublasStatus_t status, const std::source_location& loc) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[%s:%d] cuBLAS error: %d\n",
                loc.file_name(), (int)loc.line(), (int)status);
        abort();
    }
}

#define CUBLAS_CHECK(call) cublas_check_impl((call), std::source_location::current())
