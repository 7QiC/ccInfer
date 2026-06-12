#pragma once

#include <cuda_bf16.h>

#include <memory>
#include <vector>

#include "base/result.h"
#include "backend/device_buffer.h"
#include "backend/default_backend.h"
#include "model/config.h"
#include "model/loader.h"

namespace ccinfer {

struct Qwen3LayerWeights {
    std::unique_ptr<DeviceBuffer> qkv_;
    std::unique_ptr<DeviceBuffer> o_;
    std::unique_ptr<DeviceBuffer> gate_;
    std::unique_ptr<DeviceBuffer> up_;
    std::unique_ptr<DeviceBuffer> down_;
    std::unique_ptr<DeviceBuffer> rms_attn_;
    std::unique_ptr<DeviceBuffer> rms_ffn_;
    std::unique_ptr<DeviceBuffer> q_norm_;
    std::unique_ptr<DeviceBuffer> k_norm_;
};

struct Qwen3Weights {
    std::unique_ptr<DeviceBuffer> embed_;
    std::unique_ptr<DeviceBuffer> lm_head_;
    std::unique_ptr<DeviceBuffer> rms_final_;
    std::vector<Qwen3LayerWeights> layers_;

    static Result<Qwen3Weights> load(DefaultBackend& backend, const ModelConfig& config,
                                     const WeightLoader& loader);
};

}  // namespace ccinfer