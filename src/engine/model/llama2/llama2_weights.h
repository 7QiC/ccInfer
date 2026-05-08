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

struct LlamaLayerWeights {
    std::unique_ptr<DeviceBuffer> qkv_;
    std::unique_ptr<DeviceBuffer> o_;
    std::unique_ptr<DeviceBuffer> gate_;
    std::unique_ptr<DeviceBuffer> up_;
    std::unique_ptr<DeviceBuffer> down_;
    std::unique_ptr<DeviceBuffer> rms_attn_;
    std::unique_ptr<DeviceBuffer> rms_ffn_;
};

struct LlamaWeights {
    std::unique_ptr<DeviceBuffer> embed_;
    std::unique_ptr<DeviceBuffer> lm_head_;
    std::unique_ptr<DeviceBuffer> rms_final_;
    std::vector<LlamaLayerWeights> layers_;

    static Result<LlamaWeights> load(DefaultBackend& backend,
                                     const ModelConfig& config,
                                     const WeightLoader& loader);
};

}  // namespace engine
}  // namespace ccinfer
