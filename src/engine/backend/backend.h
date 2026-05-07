#pragma once

#include <memory>

#include "common/result.h"
#include "engine/backend/device_buffer.h"
#include "engine/backend/params.h"

namespace ccinfer {
namespace engine {

class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;

    virtual Result<void> gemm(const GemmParams& p) = 0;
    virtual Result<void> rms_norm(const RmsNormParams& p) = 0;
    virtual Result<void> rope(const RopeParams& p) = 0;
    virtual Result<void> silu_mul(const SiluMulParams& p) = 0;
    virtual Result<void> element_add(const ElementAddParams& p) = 0;
    virtual Result<void> split_qkv(const SplitQkvParams& p) = 0;
    virtual Result<void> naive_attention(const NaiveAttnParams& p) = 0;
    virtual Result<void> prefill_attention(const PrefillAttnParams& p) = 0;
    virtual Result<void> decode_attention(const DecodeAttnParams& p) = 0;
    virtual Result<void> write_kv_cache(const WriteKVCacheParams& p) = 0;

    virtual std::unique_ptr<DeviceBuffer> allocate_buffer(size_t bytes) = 0;

    virtual void* stream() = 0;

    static std::unique_ptr<DeviceBackend> create();
};

}  // namespace engine
}  // namespace ccinfer
