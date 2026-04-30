#include "engine/kernel/pos/rope_cache.h"

#include <cmath>
#include <vector>

#include "backend/cuda/cuda_utils.h"

namespace ccinfer {
namespace engine {

void RopeCache::init(int max_position,
                     int rotary_dim,
                     float rope_theta) {
    if (max_position <= 0) {
        return;
    }

    if (rotary_dim <= 0 || rotary_dim % 2 != 0) {
        return;
    }

    max_position_ = max_position;
    rotary_dim_ = rotary_dim;
    rope_theta_ = rope_theta;

    const int half_dim = rotary_dim_ / 2;

    std::vector<float2> host_cache(
        static_cast<size_t>(max_position_) * half_dim);

    for (int pos = 0; pos < max_position_; ++pos) {
        for (int pair = 0; pair < half_dim; ++pair) {
            const int dim_idx = pair * 2;

            const float inv_freq =
                1.0f / std::pow(
                    rope_theta_,
                    static_cast<float>(dim_idx) / static_cast<float>(rotary_dim_));

            const float angle = static_cast<float>(pos) * inv_freq;

            host_cache[static_cast<size_t>(pos) * half_dim + pair] =
                make_float2(std::cos(angle), std::sin(angle));
        }
    }

    cache_.allocate(host_cache.size());

    CCINFER_CUDA_CHECK(cudaMemcpy(
        cache_.data(),
        host_cache.data(),
        host_cache.size() * sizeof(float2),
        cudaMemcpyHostToDevice));
}

}  // namespace engine
}  // namespace ccinfer
