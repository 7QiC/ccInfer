#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "core/device_buffer.h"

namespace ccinfer {
namespace engine {

class RopeCache {
public:
    RopeCache() = default;

    void init(int max_position, int rotary_dim, float rope_theta);

    const float2* data() const noexcept { return cache_.data(); }

    float2* data() noexcept { return cache_.data(); }

    int max_position() const noexcept { return max_position_; }

    int rotary_dim() const noexcept { return rotary_dim_; }

    int half_rotary_dim() const noexcept { return rotary_dim_ / 2; }

    float rope_theta() const noexcept { return rope_theta_; }

    size_t numel() const noexcept { return static_cast<size_t>(max_position_) * half_rotary_dim(); }

    size_t bytes() const noexcept { return numel() * sizeof(float2); }

private:
    int max_position_ = 0;
    int rotary_dim_ = 0;
    float rope_theta_ = 10000.0f;

    DeviceBuffer<float2> cache_;
};

}  // namespace engine
}  // namespace ccinfer
