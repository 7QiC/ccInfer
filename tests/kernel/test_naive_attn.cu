#include <gtest/gtest.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "engine/kernel/attention/naive_attn.h"

using namespace ccinfer;
using namespace ccinfer::engine;

namespace {

// Token-major layout: [num_tokens, n_heads, head_dim]
// Element (t, h, d) = t * n_heads * head_dim + h * head_dim + d
void naive_attn_cpu(__nv_bfloat16* output, const __nv_bfloat16* q, const __nv_bfloat16* k,
                    const __nv_bfloat16* v, int num_tokens, int n_q_heads, int n_kv_heads,
                    int head_dim) {
    int gqa = n_q_heads / n_kv_heads;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    int q_stride = n_q_heads * head_dim;
    int kv_stride = n_kv_heads * head_dim;

    for (int hq = 0; hq < n_q_heads; hq++) {
        int hk = hq / gqa;
        std::vector<float> scores(num_tokens * num_tokens);

        for (int i = 0; i < num_tokens; i++) {
            for (int j = 0; j < num_tokens; j++) {
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    s += __bfloat162float(q[i * q_stride + hq * head_dim + d]) *
                         __bfloat162float(k[j * kv_stride + hk * head_dim + d]);
                scores[i * num_tokens + j] = s * scale;
            }
        }

        for (int i = 0; i < num_tokens; i++)
            for (int j = i + 1; j < num_tokens; j++)
                scores[i * num_tokens + j] = -1e30f;

        for (int i = 0; i < num_tokens; i++) {
            float max_val = -1e30f;
            for (int j = 0; j <= i; j++)
                max_val = fmaxf(max_val, scores[i * num_tokens + j]);
            float sum = 0.0f;
            for (int j = 0; j <= i; j++) {
                scores[i * num_tokens + j] = expf(scores[i * num_tokens + j] - max_val);
                sum += scores[i * num_tokens + j];
            }
            for (int j = 0; j <= i; j++)
                scores[i * num_tokens + j] /= sum;
        }

        for (int i = 0; i < num_tokens; i++) {
            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (int j = 0; j <= i; j++)
                    val += scores[i * num_tokens + j] *
                           __bfloat162float(v[j * kv_stride + hk * head_dim + d]);
                output[i * q_stride + hq * head_dim + d] = __float2bfloat16_rn(val);
            }
        }
    }
}

void fill_random(__nv_bfloat16* data, int n, unsigned int seed) {
    unsigned int s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1103515245 + 12345;
        float v = static_cast<float>((s >> 16) & 0x7FFF) / 32768.0f - 1.0f;
        data[i] = __float2bfloat16_rn(v * 0.5f);
    }
}

}  // namespace

class NaiveAttnTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override { cudaStreamDestroy(stream_); }
    cudaStream_t stream_{};
};

TEST_F(NaiveAttnTest, SingleHeadSmallSeq) {
    constexpr int tokens = 4, heads = 1, dim = 8;
    int nelts = tokens * heads * dim;

    std::vector<__nv_bfloat16> q_h(nelts), k_h(nelts), v_h(nelts), expected(nelts);
    fill_random(q_h.data(), nelts, 1);
    fill_random(k_h.data(), nelts, 2);
    fill_random(v_h.data(), nelts, 3);
    naive_attn_cpu(expected.data(), q_h.data(), k_h.data(), v_h.data(), tokens, heads, heads, dim);

    __nv_bfloat16 *q_d, *k_d, *v_d, *out_d;
    cudaMalloc(&q_d, nelts * sizeof(__nv_bfloat16));
    cudaMalloc(&k_d, nelts * sizeof(__nv_bfloat16));
    cudaMalloc(&v_d, nelts * sizeof(__nv_bfloat16));
    cudaMalloc(&out_d, nelts * sizeof(__nv_bfloat16));
    cudaMemcpy(q_d, q_h.data(), nelts * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(k_d, k_h.data(), nelts * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h.data(), nelts * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    ASSERT_TRUE(naive_attention(q_d, k_d, v_d, out_d, tokens, heads, heads, dim, stream_));
    cudaStreamSynchronize(stream_);

    std::vector<__nv_bfloat16> out_h(nelts);
    cudaMemcpy(out_h.data(), out_d, nelts * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nelts; i++)
        EXPECT_NEAR(__bfloat162float(out_h[i]), __bfloat162float(expected[i]), 0.05f);

    cudaFree(q_d); cudaFree(k_d); cudaFree(v_d); cudaFree(out_d);
}

TEST_F(NaiveAttnTest, MultiHeadGQA) {
    constexpr int tokens = 4, q_heads = 4, kv_heads = 2, dim = 8;
    int n_q = tokens * q_heads * dim;
    int n_kv = tokens * kv_heads * dim;

    std::vector<__nv_bfloat16> q_h(n_q), k_h(n_kv), v_h(n_kv), expected(n_q);
    fill_random(q_h.data(), n_q, 1);
    fill_random(k_h.data(), n_kv, 2);
    fill_random(v_h.data(), n_kv, 3);
    naive_attn_cpu(expected.data(), q_h.data(), k_h.data(), v_h.data(), tokens, q_heads, kv_heads,
                   dim);

    __nv_bfloat16 *q_d, *k_d, *v_d, *out_d;
    cudaMalloc(&q_d, n_q * sizeof(__nv_bfloat16));
    cudaMalloc(&k_d, n_kv * sizeof(__nv_bfloat16));
    cudaMalloc(&v_d, n_kv * sizeof(__nv_bfloat16));
    cudaMalloc(&out_d, n_q * sizeof(__nv_bfloat16));
    cudaMemcpy(q_d, q_h.data(), n_q * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(k_d, k_h.data(), n_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(v_d, v_h.data(), n_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    ASSERT_TRUE(naive_attention(q_d, k_d, v_d, out_d, tokens, q_heads, kv_heads, dim, stream_));
    cudaStreamSynchronize(stream_);

    std::vector<__nv_bfloat16> out_h(n_q);
    cudaMemcpy(out_h.data(), out_d, n_q * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_q; i++)
        EXPECT_NEAR(__bfloat162float(out_h[i]), __bfloat162float(expected[i]), 0.05f);

    cudaFree(q_d); cudaFree(k_d); cudaFree(v_d); cudaFree(out_d);
}
