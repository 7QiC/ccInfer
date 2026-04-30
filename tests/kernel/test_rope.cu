#include <gtest/gtest.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "kernel/pos/rope.h"
#include "kernel/pos/rope_cache.h"

using namespace ccinfer;
using namespace ccinfer::engine;

// CPU reference — split-half RoPE with rope_cache
static std::vector<float2> build_rope_cache_cpu(int max_position, int rotary_dim, float rope_theta) {
    int half_dim = rotary_dim / 2;
    std::vector<float2> cache(static_cast<size_t>(max_position) * half_dim);
    for (int pos = 0; pos < max_position; ++pos) {
        for (int pair = 0; pair < half_dim; ++pair) {
            int dim_idx = pair * 2;
            float inv_freq = 1.0f / std::pow(rope_theta,
                                              static_cast<float>(dim_idx) / static_cast<float>(rotary_dim));
            float angle = static_cast<float>(pos) * inv_freq;
            cache[static_cast<size_t>(pos) * half_dim + pair] = make_float2(std::cos(angle), std::sin(angle));
        }
    }
    return cache;
}

// CPU split-half RoPE application
static void rope_cpu(half* q, half* k, const int32_t* positions, const float2* cache,
                     int num_tokens, int num_q_heads, int num_kv_heads, int head_dim,
                     int rotary_dim, int max_position) {
    int half_dim = rotary_dim / 2;
    int total_heads = num_q_heads + num_kv_heads;
    for (int t = 0; t < num_tokens; t++) {
        int pos = positions[t];
        if (pos < 0 || pos >= max_position) continue;
        for (int h = 0; h < total_heads; h++) {
            bool is_q = h < num_q_heads;
            int head = is_q ? h : h - num_q_heads;
            int n_heads = is_q ? num_q_heads : num_kv_heads;
            half* x = is_q ? q : k;
            int64_t base = (static_cast<int64_t>(t) * n_heads + head) * head_dim;
            for (int pair = 0; pair < half_dim; pair++) {
                int i0 = pair;
                int i1 = pair + half_dim;
                float2 cs = cache[static_cast<int64_t>(pos) * half_dim + pair];
                float x0 = __half2float(x[base + i0]);
                float x1 = __half2float(x[base + i1]);
                x[base + i0] = __float2half_rn(x0 * cs.x - x1 * cs.y);
                x[base + i1] = __float2half_rn(x1 * cs.x + x0 * cs.y);
            }
        }
    }
}

class RopeTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_{};
};

TEST_F(RopeTest, SingleTokenNoGQA) {
    constexpr int tokens = 1, heads = 2, dim = 8;
    constexpr int max_pos = 16, rotary_dim = 8;

    // Build cache
    auto cache_cpu = build_rope_cache_cpu(max_pos, rotary_dim, 10000.0f);
    float2* cache_d;
    cudaMalloc(&cache_d, cache_cpu.size() * sizeof(float2));
    cudaMemcpy(cache_d, cache_cpu.data(), cache_cpu.size() * sizeof(float2), cudaMemcpyHostToDevice);

    // Input data
    int total_q = tokens * heads * dim;
    int total_k = tokens * heads * dim;
    std::vector<half> q_h(total_q), k_h(total_k), q_expected(total_q), k_expected(total_k);
    std::vector<int32_t> pos_h = {3};
    for (int i = 0; i < total_q; i++) {
        q_h[i] = __float2half((float)(i % 5));
        k_h[i] = __float2half((float)((i + 1) % 5));
    }
    q_expected = q_h;
    k_expected = k_h;
    rope_cpu(q_expected.data(), k_expected.data(), pos_h.data(), cache_cpu.data(),
             tokens, heads, heads, dim, rotary_dim, max_pos);

    half *q_d, *k_d; int32_t *pos_d;
    cudaMalloc(&q_d, total_q * sizeof(half));
    cudaMalloc(&k_d, total_k * sizeof(half));
    cudaMalloc(&pos_d, pos_h.size() * sizeof(int32_t));
    cudaMemcpy(q_d, q_h.data(), total_q * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(k_d, k_h.data(), total_k * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_d, pos_h.data(), pos_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    ASSERT_TRUE(launch_rope(q_d, k_d, pos_d, cache_d, tokens, heads, heads, dim, rotary_dim, max_pos, stream_));
    cudaStreamSynchronize(stream_);

    std::vector<half> q_out(total_q), k_out(total_k);
    cudaMemcpy(q_out.data(), q_d, total_q * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(k_out.data(), k_d, total_k * sizeof(half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < total_q; i++) {
        EXPECT_NEAR(__half2float(q_out[i]), __half2float(q_expected[i]), 0.02f);
        EXPECT_NEAR(__half2float(k_out[i]), __half2float(k_expected[i]), 0.02f);
    }

    cudaFree(q_d); cudaFree(k_d); cudaFree(pos_d); cudaFree(cache_d);
}

TEST_F(RopeTest, GQA) {
    constexpr int tokens = 2, q_heads = 8, kv_heads = 2, dim = 64;
    constexpr int max_pos = 32, rotary_dim = 64;

    auto cache_cpu = build_rope_cache_cpu(max_pos, rotary_dim, 10000.0f);
    float2* cache_d;
    cudaMalloc(&cache_d, cache_cpu.size() * sizeof(float2));
    cudaMemcpy(cache_d, cache_cpu.data(), cache_cpu.size() * sizeof(float2), cudaMemcpyHostToDevice);

    int total_q = tokens * q_heads * dim;
    int total_k = tokens * kv_heads * dim;
    std::vector<half> q_h(total_q), k_h(total_k), q_expected(total_q), k_expected(total_k);
    std::vector<int32_t> pos_h = {5, 17};
    for (int i = 0; i < total_q; i++) q_h[i] = __float2half((float)(i % 7 - 2));
    for (int i = 0; i < total_k; i++) k_h[i] = __float2half((float)((i + 3) % 9 - 3));
    q_expected = q_h;
    k_expected = k_h;
    rope_cpu(q_expected.data(), k_expected.data(), pos_h.data(), cache_cpu.data(),
             tokens, q_heads, kv_heads, dim, rotary_dim, max_pos);

    half *q_d, *k_d; int32_t *pos_d;
    cudaMalloc(&q_d, total_q * sizeof(half));
    cudaMalloc(&k_d, total_k * sizeof(half));
    cudaMalloc(&pos_d, pos_h.size() * sizeof(int32_t));
    cudaMemcpy(q_d, q_h.data(), total_q * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(k_d, k_h.data(), total_k * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_d, pos_h.data(), pos_h.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    ASSERT_TRUE(launch_rope(q_d, k_d, pos_d, cache_d, tokens, q_heads, kv_heads, dim, rotary_dim, max_pos, stream_));
    cudaStreamSynchronize(stream_);

    std::vector<half> q_out(total_q), k_out(total_k);
    cudaMemcpy(q_out.data(), q_d, total_q * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(k_out.data(), k_d, total_k * sizeof(half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < total_q; i++)
        EXPECT_NEAR(__half2float(q_out[i]), __half2float(q_expected[i]), 0.02f);
    for (int i = 0; i < total_k; i++)
        EXPECT_NEAR(__half2float(k_out[i]), __half2float(k_expected[i]), 0.02f);

    cudaFree(q_d); cudaFree(k_d); cudaFree(pos_d); cudaFree(cache_d);
}

TEST_F(RopeTest, RopeCacheClass) {
    RopeCache cache;
    cache.init(16, 32, 10000.0f);
    EXPECT_EQ(cache.max_position(), 16);
    EXPECT_EQ(cache.rotary_dim(), 32);
    EXPECT_EQ(cache.half_rotary_dim(), 16);
    EXPECT_NE(cache.data(), nullptr);
    EXPECT_EQ(cache.numel(), 256u);
}
