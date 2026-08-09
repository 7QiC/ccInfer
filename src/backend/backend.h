#pragma once

#include <cstddef>
#include <memory>

#include "ops/ops.h"
#include "ops/ops.h"
#include "ops/ops.h"

#include "backend/buffer.h"
#include "backend/params.h"
#include "base/result.h"

namespace ccinfer {

// Framework execution-entry: a single plain Backend class (PIMPL).
//
// This header is hardware-agnostic (no CUDA/NPU types). Concrete
// implementations live in backend/<vendor>/backend_<vendor>.cpp (currently
// backend/cuda/backend_cuda.cpp) and are selected at build time by CMake.
// Memory ownership stays framework-side via Buffer; the backend never leaks
// device types into framework code.
class Backend final {
public:
    static Result<std::unique_ptr<Backend>> create(int device_id);

    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    Backend(Backend&&) noexcept;
    Backend& operator=(Backend&&) noexcept;

    Result<std::shared_ptr<Buffer>> allocate_buffer(std::size_t bytes);
    Result<void> memcpy_h2d(void* dst, const void* src, std::size_t count);
    Result<void> memcpy_d2h(void* dst, const void* src, std::size_t count);
    Result<void> memcpy_d2d(void* dst, const void* src, std::size_t count);
    [[nodiscard]] void* stream() const noexcept;
    Result<void> synchronize();
    [[nodiscard]] ops::ExecutionContext context() const noexcept;

    // Transitional explicit runtime dtype; final form will take ops::Tensor +
    // ExecutionContext once ops migrate into the operator library.
    Result<void> embed(const EmbedParams& p);
    Result<void> gemm(ops::DType dtype, const GemmParams& p);
    Result<void> gemm_logits(const GemmParams& p);
    Result<void> rms_norm(ops::DType dtype, const RmsNormParams& p);
    Result<void> rope(ops::DType dtype, const RopeParams& p);
    Result<void> silu_mul(ops::DType dtype, const SiluMulParams& p);
    Result<void> element_add(ops::DType dtype, const ElementAddParams& p);
    Result<void> split_qkv(ops::DType dtype, const SplitQkvParams& p);
    Result<void> naive_attention(ops::DType dtype, const NaiveAttnParams& p);
    Result<void> prefill_attention(ops::DType dtype, const PrefillAttnParams& p);
    Result<void> decode_attention(ops::DType dtype, const DecodeAttnParams& p);
    Result<void> write_kv_cache(ops::DType dtype, const WriteKVCacheParams& p);
    Result<void> sample(const SampleParams& p);

private:
    struct Impl;
    explicit Backend(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace ccinfer
