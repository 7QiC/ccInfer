#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "kernel/cuda_kernels.h"

using namespace ccinfer;

class SampleKernelTest : public ::testing::Test {
protected:
    void SetUp() override { cudaStreamCreate(&stream_); }
    void TearDown() override {
        ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        cudaStreamDestroy(stream_);
    }
    cudaStream_t stream_{};
};

TEST_F(SampleKernelTest, SingleSequenceGreedy) {
    constexpr int batch_size = 1;
    constexpr int num_tokens = 3;
    constexpr int vocab_size = 8;

    // 3 tokens for one sequence, argmax at position 5 for last token
    std::vector<float> logits_host(num_tokens * vocab_size, 0.0f);
    logits_host[2 * vocab_size + 5] = 10.0f;  // last token, vocab idx 5 = max
    logits_host[2 * vocab_size + 3] = 3.0f;

    std::vector<int32_t> indices_host = {2};  // last token index for seq 0

    float* logits_d;
    int32_t *tokens_d, *indices_d;
    cudaMalloc(&logits_d, num_tokens * vocab_size * sizeof(float));
    cudaMalloc(&tokens_d, batch_size * sizeof(int32_t));
    cudaMalloc(&indices_d, batch_size * sizeof(int32_t));
    cudaMemcpy(logits_d, logits_host.data(), num_tokens * vocab_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(indices_d, indices_host.data(), batch_size * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    ASSERT_TRUE(launch_greedy_sample(logits_d, tokens_d, indices_d, batch_size, vocab_size, num_tokens, stream_));

    std::vector<int32_t> tokens_host(batch_size);
    cudaMemcpy(tokens_host.data(), tokens_d, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

    EXPECT_EQ(tokens_host[0], 5);

    cudaFree(logits_d);
    cudaFree(tokens_d);
    cudaFree(indices_d);
}

TEST_F(SampleKernelTest, MultiSequence) {
    constexpr int batch_size = 3;
    constexpr int vocab_size = 16;
    // seq 0: 2 tokens → last idx 1, seq 1: 1 token → last idx 2, seq 2: 4 tokens → last idx 6
    std::vector<int32_t> indices_host = {1, 2, 6};
    int num_tokens = 7;  // 2 + 1 + 4

    std::vector<float> logits_host(num_tokens * vocab_size);
    // seq 0 last token at idx 1: set vocab[7] as max
    logits_host[1 * vocab_size + 7] = 100.0f;
    // seq 1 only token at idx 2: set vocab[3] as max
    logits_host[2 * vocab_size + 3] = 100.0f;
    // seq 2 last token at idx 6: set vocab[11] as max
    logits_host[6 * vocab_size + 11] = 100.0f;

    float* logits_d;
    int32_t *tokens_d, *indices_d;
    cudaMalloc(&logits_d, num_tokens * vocab_size * sizeof(float));
    cudaMalloc(&tokens_d, batch_size * sizeof(int32_t));
    cudaMalloc(&indices_d, batch_size * sizeof(int32_t));
    cudaMemcpy(logits_d, logits_host.data(), num_tokens * vocab_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(indices_d, indices_host.data(), batch_size * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    ASSERT_TRUE(launch_greedy_sample(logits_d, tokens_d, indices_d, batch_size, vocab_size, num_tokens, stream_));

    std::vector<int32_t> tokens_host(batch_size);
    cudaMemcpy(tokens_host.data(), tokens_d, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

    EXPECT_EQ(tokens_host[0], 7);
    EXPECT_EQ(tokens_host[1], 3);
    EXPECT_EQ(tokens_host[2], 11);

    cudaFree(logits_d);
    cudaFree(tokens_d);
    cudaFree(indices_d);
}

TEST_F(SampleKernelTest, ArgmaxWithTies) {
    constexpr int batch_size = 1;
    constexpr int vocab_size = 4;

    std::vector<float> logits_host(vocab_size);
    logits_host[0] = 5.0f;
    logits_host[1] = 0.0f;
    logits_host[2] = 5.0f;  // tie with idx 0
    logits_host[3] = 1.0f;

    std::vector<int32_t> indices_host = {0};  // single token at index 0
    constexpr int num_tokens = 1;

    float* logits_d;
    int32_t *tokens_d, *indices_d;
    cudaMalloc(&logits_d, vocab_size * sizeof(float));
    cudaMalloc(&tokens_d, batch_size * sizeof(int32_t));
    cudaMalloc(&indices_d, batch_size * sizeof(int32_t));
    cudaMemcpy(logits_d, logits_host.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(indices_d, indices_host.data(), batch_size * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    ASSERT_TRUE(launch_greedy_sample(logits_d, tokens_d, indices_d, batch_size, vocab_size, num_tokens, stream_));

    std::vector<int32_t> tokens_host(batch_size);
    cudaMemcpy(tokens_host.data(), tokens_d, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Tie-breaker: lower index wins.
    EXPECT_EQ(tokens_host[0], 0);

    cudaFree(logits_d);
    cudaFree(tokens_d);
    cudaFree(indices_d);
}

TEST_F(SampleKernelTest, LargeVocab) {
    constexpr int batch_size = 1;
    constexpr int vocab_size = 32000;  // realistic vocab

    std::vector<float> logits_host(vocab_size, -1.0f);
    logits_host[12345] = 50.0f;  // max

    std::vector<int32_t> indices_host = {0};  // single token at index 0
    constexpr int num_tokens = 1;

    float* logits_d;
    int32_t *tokens_d, *indices_d;
    cudaMalloc(&logits_d, vocab_size * sizeof(float));
    cudaMalloc(&tokens_d, batch_size * sizeof(int32_t));
    cudaMalloc(&indices_d, batch_size * sizeof(int32_t));
    cudaMemcpy(logits_d, logits_host.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(indices_d, indices_host.data(), batch_size * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    ASSERT_TRUE(launch_greedy_sample(logits_d, tokens_d, indices_d, batch_size, vocab_size, num_tokens, stream_));

    std::vector<int32_t> tokens_host(batch_size);
    cudaMemcpy(tokens_host.data(), tokens_d, batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

    EXPECT_EQ(tokens_host[0], 12345);

    cudaFree(logits_d);
    cudaFree(tokens_d);
    cudaFree(indices_d);
}
