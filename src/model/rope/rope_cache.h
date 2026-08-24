#pragma once

#include <cstddef>
#include <memory>

#include "common/error_code.h"
#include "core/tensor.h"

namespace ccinfer {

class Backend;

class RopeCache {
public:
    static Result<RopeCache> create(int max_position, int rotary_dim, float rope_theta,
                                    Backend& backend);

    // 框架 Tensor 视图（共享 cache_ 的所有权）。
    Tensor tensor() const;

    int max_position() const noexcept { return max_position_; }
    int rotary_dim() const noexcept { return rotary_dim_; }
    int half_rotary_dim() const noexcept { return rotary_dim_ / 2; }
    float rope_theta() const noexcept { return rope_theta_; }

    // float 元素个数（cos/sin 各占一个）。
    std::size_t numel() const noexcept {
        return static_cast<std::size_t>(max_position_) * half_rotary_dim() * 2;
    }
    std::size_t bytes() const noexcept { return numel() * sizeof(float); }

private:
    RopeCache() = default;

    int max_position_ = 0;
    int rotary_dim_ = 0;
    float rope_theta_ = 10000.0f;

    std::shared_ptr<Buffer> cache_;
};

}  // namespace ccinfer
