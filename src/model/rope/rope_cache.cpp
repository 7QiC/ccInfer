#include "model/rope/rope_cache.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "backend/backend.h"

namespace ccinfer {

Result<RopeCache> RopeCache::create(int max_position, int rotary_dim, float rope_theta,
                                    Backend& backend) {
    if (max_position <= 0 || rotary_dim <= 0 || rotary_dim % 2 != 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    if (rope_theta <= 0.0f || !std::isfinite(rope_theta)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    const std::size_t floats_per_pos = static_cast<std::size_t>(rotary_dim);
    if (static_cast<std::size_t>(max_position) >
        std::numeric_limits<std::size_t>::max() / floats_per_pos) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }
    const std::size_t total_floats = static_cast<std::size_t>(max_position) * floats_per_pos;

    RopeCache cache;
    cache.max_position_ = max_position;
    cache.rotary_dim_ = rotary_dim;
    cache.rope_theta_ = rope_theta;

    const int half_dim = rotary_dim / 2;
    std::vector<float> inv_freqs(static_cast<std::size_t>(half_dim));
    for (int pair = 0; pair < half_dim; ++pair) {
        inv_freqs[static_cast<std::size_t>(pair)] =
            1.0f /
            std::pow(rope_theta, static_cast<float>(2 * pair) / static_cast<float>(rotary_dim));
    }

    std::vector<float> host_cache(total_floats);
    for (int pos = 0; pos < max_position; ++pos) {
        const float fpos = static_cast<float>(pos);
        for (int pair = 0; pair < half_dim; ++pair) {
            const float angle = fpos * inv_freqs[static_cast<std::size_t>(pair)];
            const std::size_t idx = static_cast<std::size_t>((pos * half_dim + pair) * 2);
            host_cache[idx + 0] = std::cos(angle);
            host_cache[idx + 1] = std::sin(angle);
        }
    }

    auto buf = backend.allocate_buffer(host_cache.size() * sizeof(float));
    if (!buf) return std::unexpected(buf.error());
    cache.cache_ = std::move(*buf);

    auto r = backend.memcpy_h2d(cache.cache_->data(), host_cache.data(),
                                host_cache.size() * sizeof(float));
    if (!r) return std::unexpected(r.error());

    auto sync_r = backend.synchronize();
    if (!sync_r) return std::unexpected(sync_r.error());

    return cache;
}

Tensor RopeCache::tensor() const {
    return Tensor(cache_, ccop::DType::kFloat32,
                  {static_cast<std::int64_t>(max_position_),
                   static_cast<std::int64_t>(half_rotary_dim()), 2});
}

}  // namespace ccinfer
