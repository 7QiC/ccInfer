#pragma once

#include <cstddef>
#include <memory>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "common/result.h"
#include "engine/backend/backend.h"
#include "engine/backend/device_buffer.h"
#include "engine/backend/params.h"

namespace ccinfer {
namespace engine {

class CudaBackend final : public DeviceBackend<CudaBackend> {
public:
    CudaBackend() = default;
    ~CudaBackend();

    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    CudaBackend(CudaBackend&&) noexcept;
    CudaBackend& operator=(CudaBackend&&) noexcept;

    Result<void> init(int device_id);

    template <typename DType>
    Result<void> gemm_impl(const GemmParams& p);

    template <typename DType>
    Result<void> rms_norm_impl(const RmsNormParams& p);

    template <typename DType>
    Result<void> rope_impl(const RopeParams& p);

    template <typename DType>
    Result<void> silu_mul_impl(const SiluMulParams& p);

    template <typename DType>
    Result<void> element_add_impl(const ElementAddParams& p);

    template <typename DType>
    Result<void> split_qkv_impl(const SplitQkvParams& p);

    template <typename DType>
    Result<void> naive_attention_impl(const NaiveAttnParams& p);

    template <typename DType>
    Result<void> prefill_attention_impl(const PrefillAttnParams& p);

    template <typename DType>
    Result<void> decode_attention_impl(const DecodeAttnParams& p);

    template <typename DType>
    Result<void> write_kv_cache_impl(const WriteKVCacheParams& p);

    std::unique_ptr<DeviceBuffer> allocate_buffer_impl(std::size_t bytes);

    Result<void> memcpy_h2d_impl(void* dst, const void* src, std::size_t count);
    Result<void> memcpy_d2h_impl(void* dst, const void* src, std::size_t count);
    Result<void> memcpy_d2d_impl(void* dst, const void* src, std::size_t count);

    void* stream_impl();

private:
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
};

}  // namespace engine
}  // namespace ccinfer
