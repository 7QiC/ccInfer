#include "model/rope/rope_cache.h"

#include <cmath>
#include <utility>
#include <vector>

namespace ccinfer {

Result<RopeCache> RopeCache::create(int max_position, int rotary_dim, float rope_theta,
                                    DefaultBackend& backend) {
    if (max_position <= 0 || rotary_dim <= 0 || rotary_dim % 2 != 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (rope_theta <= 0.0f || !std::isfinite(rope_theta)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    const int half_dim = rotary_dim / 2;
    const std::size_t total_elems =
        static_cast<std::size_t>(max_position) * static_cast<std::size_t>(half_dim);
    if (half_dim != 0 && total_elems / static_cast<std::size_t>(half_dim) !=
                             static_cast<std::size_t>(max_position)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    RopeCache cache;
    cache.max_position_ = max_position;
    cache.rotary_dim_ = rotary_dim;
    cache.rope_theta_ = rope_theta;

    std::vector<float> inv_freqs(half_dim);
    for (int pair = 0; pair < half_dim; ++pair) {
        inv_freqs[pair] = 1.0f / std::pow(rope_theta, static_cast<float>(pair * 2) /
                                                          static_cast<float>(rotary_dim));
    }

    std::vector<float2> host_cache(total_elems);
    for (int pos = 0; pos < max_position; ++pos) {
        const float fpos = static_cast<float>(pos);
        for (int pair = 0; pair < half_dim; ++pair) {
            const float angle = fpos * inv_freqs[pair];
            host_cache[static_cast<std::size_t>(pos) * static_cast<std::size_t>(half_dim) + pair] =
                make_float2(std::cos(angle), std::sin(angle));
        }
    }

    auto buf = backend.allocate_buffer(host_cache.size() * sizeof(float2));
    if (!buf) return std::unexpected(buf.error());
    cache.cache_ = std::move(*buf);

    auto r = backend.memcpy_h2d(cache.cache_->data(), host_cache.data(),
                                host_cache.size() * sizeof(float2));
    if (!r) return std::unexpected(r.error());

    auto sync_r = backend.synchronize();
    if (!sync_r) return std::unexpected(sync_r.error());

    return std::move(cache);
}

}  // namespace ccinfer
