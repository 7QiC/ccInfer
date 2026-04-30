#pragma once

#include <cuda_bf16.h>

#include <vector>

#include "engine/core/device_buffer.h"

namespace ccinfer {
namespace engine {

struct LayerWeights {
    DeviceBuffer<__nv_bfloat16> qkv_;
    DeviceBuffer<__nv_bfloat16> o_;
    DeviceBuffer<__nv_bfloat16> gate_;
    DeviceBuffer<__nv_bfloat16> up_;
    DeviceBuffer<__nv_bfloat16> down_;
    DeviceBuffer<__nv_bfloat16> rms_attn_;
    DeviceBuffer<__nv_bfloat16> rms_ffn_;
};

struct ModelWeights {
    DeviceBuffer<__nv_bfloat16> embed_;
    DeviceBuffer<__nv_bfloat16> lm_head_;
    DeviceBuffer<__nv_bfloat16> rms_final_;
    std::vector<LayerWeights> layers_;
};

}  // namespace engine
}  // namespace ccinfer
