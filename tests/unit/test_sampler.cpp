#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "engine/sampling/sampler.h"

using namespace ccinfer::engine;

TEST(SamplerTest, DeterministicWithSeed) {
    Sampler sampler(42);

    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 0;
    params.top_p = 1.0f;

    float logits[] = {0.1f, 0.2f, 0.3f, 5.0f, 0.1f, 0.05f, 0.15f, 0.0f};
    int n = sizeof(logits) / sizeof(logits[0]);

    auto t1 = sampler.sample_token(logits, n, params);
    auto t2 = sampler.sample_token(logits, n, params);
    ASSERT_TRUE(t1);
    ASSERT_TRUE(t2);
    EXPECT_EQ(*t1, *t2);
}

TEST(SamplerTest, TemperatureZeroIsGreedy) {
    Sampler sampler(100);

    SamplingParams params;
    params.temperature = 0.0f;

    float logits[] = {0.0f, 1.0f, 2.0f, 0.5f};
    int n = sizeof(logits) / sizeof(logits[0]);

    auto t = sampler.sample_token(logits, n, params);
    ASSERT_TRUE(t);
    EXPECT_EQ(*t, 2);  // argmax
}

TEST(SamplerTest, TopKOneIsGreedy) {
    Sampler sampler(7);

    SamplingParams params;
    params.temperature = 1.0f;
    params.top_k = 1;
    params.top_p = 1.0f;

    float logits[] = {1.0f, 5.0f, 3.0f, 2.0f};
    int n = sizeof(logits) / sizeof(logits[0]);

    for (int i = 0; i < 10; ++i) {
        auto t = sampler.sample_token(logits, n, params);
        ASSERT_TRUE(t);
        EXPECT_EQ(*t, 1);  // Always index 1 (value 5.0)
    }
}

TEST(SamplerTest, OutputInRange) {
    Sampler sampler(12345);

    SamplingParams params;
    params.temperature = 0.8f;
    params.top_k = 50;
    params.top_p = 0.9f;

    constexpr int vocab = 128;
    std::vector<float> logits(vocab);
    for (int i = 0; i < vocab; ++i) {
        logits[static_cast<size_t>(i)] = std::sin(static_cast<float>(i) * 0.1f);
    }

    for (int i = 0; i < 20; ++i) {
        sampler.reset_seed(static_cast<unsigned int>(i * 100));
        auto t = sampler.sample_token(logits.data(), vocab, params);
        ASSERT_TRUE(t);
        EXPECT_GE(*t, 0);
        EXPECT_LT(*t, vocab);
    }
}

TEST(SamplerTest, NullLogitsReturnsError) {
    Sampler sampler(0);
    SamplingParams params;
    auto t = sampler.sample_token(nullptr, 10, params);
    EXPECT_FALSE(t);
}

TEST(SamplerTest, ZeroVocabReturnsError) {
    Sampler sampler(0);
    SamplingParams params;
    float logits[] = {1.0f};
    auto t = sampler.sample_token(logits, 0, params);
    EXPECT_FALSE(t);
}
