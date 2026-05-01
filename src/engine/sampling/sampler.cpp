#include "engine/sampling/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "common/error_code.h"

namespace ccinfer {
namespace engine {

namespace {

int32_t argmax_token(const float* logits, int vocab_size) {
    int32_t best = 0;
    float best_val = logits[0];

    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }

    return best;
}

bool is_valid_float(float x) { return std::isfinite(x); }

}  // namespace

Result<int32_t> Sampler::sample_token(const float* logits, int vocab_size,
                                      const SamplingParams& params) {
    if (logits == nullptr || vocab_size <= 0) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    // temperature <= 0 means greedy.
    if (params.temperature <= 0.0f || params.top_k == 1) {
        return argmax_token(logits, vocab_size);
    }

    const float temperature = params.temperature;

    std::vector<float> filtered(static_cast<size_t>(vocab_size));

    // Step 1: temperature scaling.
    for (int i = 0; i < vocab_size; ++i) {
        const float x = logits[i];

        if (!is_valid_float(x)) {
            filtered[static_cast<size_t>(i)] = -std::numeric_limits<float>::infinity();
        } else {
            filtered[static_cast<size_t>(i)] = x / temperature;
        }
    }

    // Step 2: top-k filtering.
    const int top_k = params.top_k;

    if (top_k > 0 && top_k < vocab_size) {
        std::vector<int32_t> idx(static_cast<size_t>(vocab_size));
        std::iota(idx.begin(), idx.end(), 0);

        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [&](int32_t a, int32_t b) {
            return filtered[static_cast<size_t>(a)] > filtered[static_cast<size_t>(b)];
        });

        std::vector<float> topk_filtered(static_cast<size_t>(vocab_size),
                                         -std::numeric_limits<float>::infinity());

        for (int i = 0; i < top_k; ++i) {
            const int32_t token = idx[static_cast<size_t>(i)];
            topk_filtered[static_cast<size_t>(token)] = filtered[static_cast<size_t>(token)];
        }

        filtered = std::move(topk_filtered);
    }

    // Step 3: softmax.
    float max_logit = -std::numeric_limits<float>::infinity();

    for (float x : filtered) {
        if (x > max_logit) {
            max_logit = x;
        }
    }

    if (!std::isfinite(max_logit)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    std::vector<float> probs(static_cast<size_t>(vocab_size));
    float sum = 0.0f;

    for (int i = 0; i < vocab_size; ++i) {
        const float x = filtered[static_cast<size_t>(i)];

        if (std::isfinite(x)) {
            const float p = std::exp(x - max_logit);
            probs[static_cast<size_t>(i)] = p;
            sum += p;
        } else {
            probs[static_cast<size_t>(i)] = 0.0f;
        }
    }

    if (sum <= 0.0f || !std::isfinite(sum)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    for (float& p : probs) {
        p /= sum;
    }

    // Step 4: top-p filtering.
    const float top_p = params.top_p;

    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<int32_t> idx(static_cast<size_t>(vocab_size));
        std::iota(idx.begin(), idx.end(), 0);

        std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
            return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)];
        });

        float cumulative = 0.0f;
        int keep_count = 0;

        for (int i = 0; i < vocab_size; ++i) {
            const int32_t token = idx[static_cast<size_t>(i)];
            cumulative += probs[static_cast<size_t>(token)];
            keep_count = i + 1;

            if (cumulative >= top_p) {
                break;
            }
        }

        // Always keep at least one token.
        if (keep_count <= 0) {
            keep_count = 1;
        }

        std::vector<float> top_p_probs(static_cast<size_t>(vocab_size), 0.0f);

        for (int i = 0; i < keep_count; ++i) {
            const int32_t token = idx[static_cast<size_t>(i)];
            top_p_probs[static_cast<size_t>(token)] = probs[static_cast<size_t>(token)];
        }

        float p_sum = 0.0f;
        for (float p : top_p_probs) {
            p_sum += p;
        }

        if (p_sum <= 0.0f || !std::isfinite(p_sum)) {
            return std::unexpected(ErrorCode::InvalidArgument);
        }

        for (float& p : top_p_probs) {
            p /= p_sum;
        }

        probs = std::move(top_p_probs);
    }

    // Step 5: multinomial sampling.
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float r = dist(rng_);

    float cumulative = 0.0f;

    for (int i = 0; i < vocab_size; ++i) {
        cumulative += probs[static_cast<size_t>(i)];

        if (r < cumulative) {
            return static_cast<int32_t>(i);
        }
    }

    // Floating-point fallback.
    return static_cast<int32_t>(std::max_element(probs.begin(), probs.end()) - probs.begin());
}

}  // namespace engine
}  // namespace ccinfer
