#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "engine/backend/backend.h"

namespace ccinfer {
namespace engine {

class CudaBackend : public DeviceBackend {
public:
    CudaBackend();
    ~CudaBackend() override;

    Result<void> gemm(const GemmParams& p) override;
    Result<void> rms_norm(const RmsNormParams& p) override;
    Result<void> rope(const RopeParams& p) override;
    Result<void> silu_mul(const SiluMulParams& p) override;
    Result<void> element_add(const ElementAddParams& p) override;
    Result<void> split_qkv(const SplitQkvParams& p) override;
    Result<void> naive_attention(const NaiveAttnParams& p) override;
    Result<void> prefill_attention(const PrefillAttnParams& p) override;
    Result<void> decode_attention(const DecodeAttnParams& p) override;
    Result<void> write_kv_cache(const WriteKVCacheParams& p) override;

private:
    cublasHandle_t cublas_handle_;
    cudaStream_t stream_;
};

}  // namespace engine
}  // namespace ccinfer
