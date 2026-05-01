#pragma once

#include <cstdint>
#include <random>

#include "common/result.h"

namespace ccinfer {
namespace engine {

struct SamplingParams {
    int top_k = 50;            // 0 means disabled
    float top_p = 0.9f;        // >= 1 means disabled
    float temperature = 1.0f;  // <= 0 means greedy
    uint32_t seed = 42;
};

class Sampler {
public:
    explicit Sampler(uint32_t seed = 42) : rng_(seed) {}

    void reset_seed(uint32_t seed) { rng_.seed(seed); }

    // logits: host-side [vocab_size] in FP32.
    Result<int32_t> sample_token(const float* logits, int vocab_size, const SamplingParams& params);

private:
    std::mt19937 rng_;
};

}  // namespace engine
}  // namespace ccinfer
