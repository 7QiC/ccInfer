#include "cuda_backend.h"
#include "cuda_utils.h"
#include "result.h"
#include <cuda_fp16.h>

namespace ccinfer {

CudaBackend::CudaBackend() {
    CCINFER_CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CCINFER_CUDA_CHECK(cudaStreamCreate(&stream_));
}

CudaBackend::~CudaBackend() {
    cublasDestroy(cublas_handle_);
    cudaStreamDestroy(stream_);
}

Result<void> CudaBackend::gemm(const GemmParams& p) {
    auto s = static_cast<cudaStream_t>(p.stream);
    CCINFER_CUBLAS_CHECK(cublasSetStream(cublas_handle_, s));

    half alpha = __float2half(1.0f);
    half beta  = __float2half(0.0f);

    CCINFER_CUBLAS_CHECK(cublasGemmEx(
        cublas_handle_,
        p.trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
        p.trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
        p.n, p.m, p.k,
        &alpha,
        p.B, CUDA_R_16F, p.ldb,
        p.A, CUDA_R_16F, p.lda,
        &beta,
        p.C, CUDA_R_16F, p.ldc,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return {};
}

Result<void> CudaBackend::rms_norm(const RmsNormParams&) { return {}; }
Result<void> CudaBackend::rope(const RopeParams&) { return {}; }
Result<void> CudaBackend::silu_mul(const SiluMulParams&) { return {}; }

std::unique_ptr<DeviceBackend> DeviceBackend::create() {
    return std::make_unique<CudaBackend>();
}

} // namespace ccinfer
