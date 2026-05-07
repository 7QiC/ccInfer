#include "engine/model/rope/rope_cache.h"

#include <cmath>
#include <vector>

#include "engine/backend/backend.h"
#include "engine/backend/cuda/cuda_utils.h"

namespace ccinfer {
namespace engine {

Result<RopeCache> RopeCache::create(int max_position, int rotary_dim, float rope_theta,
                                    DeviceBackend& backend) {
    if (max_position <= 0 || rotary_dim <= 0 || rotary_dim % 2 != 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    RopeCache cache;
    cache.max_position_ = max_position;
    cache.rotary_dim_ = rotary_dim;
    cache.rope_theta_ = rope_theta;

    const int half_dim = rotary_dim / 2;
    std::vector<float2> host_cache(static_cast<size_t>(max_position) * half_dim);

    for (int pos = 0; pos < max_position; ++pos) {
        for (int pair = 0; pair < half_dim; ++pair) {
            const float inv_freq = 1.0f / std::pow(rope_theta, static_cast<float>(pair * 2) /
                                                                   static_cast<float>(rotary_dim));
            const float angle = static_cast<float>(pos) * inv_freq;
            host_cache[static_cast<size_t>(pos) * half_dim + pair] =
                make_float2(std::cos(angle), std::sin(angle));
        }
    }

    cache.cache_ = backend.allocate_buffer(host_cache.size() * sizeof(float2));

    auto r = cuda_check(cudaMemcpy(cache.cache_->data(), host_cache.data(),
                                   host_cache.size() * sizeof(float2), cudaMemcpyHostToDevice));
    if (!r) return std::unexpected(r.error());

    return cache;
}

}  // namespace engine
}  // namespace ccinfer
