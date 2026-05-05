#pragma once

#include <cuda_bf16.h>

#include <vector>

#include "common/result.h"
#include "engine/core/device_buffer.h"
#include "engine/model/config.h"
#include "engine/model/loader.h"

namespace ccinfer {
namespace engine {

struct Qwen3LayerWeights {
    DeviceBuffer<__nv_bfloat16> qkv_;
    DeviceBuffer<__nv_bfloat16> o_;
    DeviceBuffer<__nv_bfloat16> gate_;
    DeviceBuffer<__nv_bfloat16> up_;
    DeviceBuffer<__nv_bfloat16> down_;
    DeviceBuffer<__nv_bfloat16> rms_attn_;
    DeviceBuffer<__nv_bfloat16> rms_ffn_;
    DeviceBuffer<__nv_bfloat16> q_norm_;
    DeviceBuffer<__nv_bfloat16> k_norm_;
};

struct Qwen3Weights {
    DeviceBuffer<__nv_bfloat16> embed_;
    DeviceBuffer<__nv_bfloat16> lm_head_;
    DeviceBuffer<__nv_bfloat16> rms_final_;
    std::vector<Qwen3LayerWeights> layers_;

    static Result<Qwen3Weights> load(const ModelConfig& config, const WeightLoader& loader);
};

}  // namespace engine
}  // namespace ccinfer
