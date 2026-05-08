#pragma once

#include <cuda_bf16.h>

#include <memory>
#include <vector>

#include "common/result.h"
#include "engine/backend/device_buffer.h"
#include "engine/common/backend_def.h"
#include "engine/model/config.h"
#include "engine/model/loader.h"

namespace ccinfer {
namespace engine {

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

    static Result<Qwen3Weights> load(DefaultBackend& backend,
                                     const ModelConfig& config,
                                     const WeightLoader& loader);
};

}  // namespace engine
}  // namespace ccinfer