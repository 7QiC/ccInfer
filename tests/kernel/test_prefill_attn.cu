#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <numeric>
#include <vector>

#include "engine/backend/cuda/cuda_backend.h"
#include "engine/cache/block.h"
#include "engine/kernel/cuda_kernels.h"

namespace ccinfer {
namespace engine {
namespace {

void fill_random_bf16(__nv_bfloat16* d_data, int64_t n, cudaStream_t stream) {
    std::vector<__nv_bfloat16> h(n);
    for (int64_t i = 0; i < n; ++i) {
        h[i] = __float2bfloat16((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f);
    }
    cudaMemcpyAsync(d_data, h.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream);
}

TEST(PrefillAttnKernelTest, SingleRequestMatchesNaive) {
    constexpr int kNumTokens = 32;
    constexpr int kNumQHeads = 4;
    constexpr int kNumKVHeads = 4;
    constexpr int kHeadDim = 64;
    constexpr int kMaxBlocks = 4;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int64_t q_elems = static_cast<int64_t>(kNumTokens) * kNumQHeads * kHeadDim;
    int64_t kv_elems = static_cast<int64_t>(kMaxBlocks) * kKVBlockSize * kNumKVHeads * kHeadDim;

    __nv_bfloat16 *d_q, *d_k_cache, *d_v_cache, *d_out_prefill, *d_out_naive;
    cudaMalloc(&d_q, q_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_k_cache, kv_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_v_cache, kv_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out_prefill, q_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out_naive, q_elems * sizeof(__nv_bfloat16));

    fill_random_bf16(d_q, q_elems, stream);
    fill_random_bf16(d_k_cache, kv_elems, stream);
    fill_random_bf16(d_v_cache, kv_elems, stream);

    // Run naive attention: K/V are already in contiguous cache, use as regular dense tensor
    {
        auto r = launch_naive_attention(d_q, d_k_cache, d_v_cache, d_out_naive,
                                        kNumTokens, kNumQHeads, kNumKVHeads,
                                        kHeadDim, stream);
        ASSERT_TRUE(r.has_value());
    }

    // block_table: logical blocks 0,1 map to physical blocks 0,1 (identity)
    std::vector<int32_t> h_query_start_loc = {0, kNumTokens};
    std::vector<int32_t> h_context_lens = {kNumTokens};
    std::vector<int32_t> h_block_table = {0, 1};

    int32_t *d_query_start_loc, *d_context_lens, *d_block_table;
    cudaMalloc(&d_query_start_loc, 2 * sizeof(int32_t));
    cudaMalloc(&d_context_lens, 1 * sizeof(int32_t));
    cudaMalloc(&d_block_table, 2 * sizeof(int32_t));

    cudaMemcpyAsync(d_query_start_loc, h_query_start_loc.data(),
                    2 * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_context_lens, h_context_lens.data(),
                    sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_block_table, h_block_table.data(),
                    2 * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    // Run prefill attention (reads K/V through block table)
    auto r = launch_prefill_attention(d_q, d_k_cache, d_v_cache,
                                      d_block_table, d_query_start_loc, d_context_lens,
                                      d_out_prefill, 1, kNumTokens, 2, kNumQHeads, kNumKVHeads,
                                      kHeadDim, kKVBlockSize, stream);
    ASSERT_TRUE(r.has_value());
    cudaStreamSynchronize(stream);

    // Compare outputs (bitwise, since same algorithm on same data)
    std::vector<__nv_bfloat16> h_prefill(q_elems);
    std::vector<__nv_bfloat16> h_naive(q_elems);
    cudaMemcpy(h_prefill.data(), d_out_prefill, q_elems * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_naive.data(), d_out_naive, q_elems * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (int64_t i = 0; i < q_elems; ++i) {
        float diff = fabsf(__bfloat162float(h_prefill[i]) - __bfloat162float(h_naive[i]));
        if (diff > max_diff) max_diff = diff;
    }
    EXPECT_LT(max_diff, 0.02f);

    cudaFree(d_q); cudaFree(d_k_cache); cudaFree(d_v_cache);
    cudaFree(d_out_prefill); cudaFree(d_out_naive);
    cudaFree(d_query_start_loc); cudaFree(d_context_lens);
    cudaFree(d_block_table);
    auto sync_err = cudaStreamSynchronize(stream);
    ASSERT_EQ(sync_err, cudaSuccess);
    auto last_err = cudaGetLastError();
    ASSERT_EQ(last_err, cudaSuccess);
    cudaStreamDestroy(stream);
}

TEST(PrefillAttnKernelTest, RejectsNullPointers) {
    CudaBackend backend;

    int32_t dummy_start_loc[2] = {0, 0};
    auto r = backend.template prefill_attention<__nv_bfloat16>(PrefillAttnParams{
        .q_ = nullptr, .k_cache_ = nullptr, .v_cache_ = nullptr, .block_table_ = nullptr,
        .query_start_loc_ = dummy_start_loc, .context_lens_ = nullptr, .output_ = nullptr,
        .batch_size_ = 1, .max_blocks_per_req_ = 1, .num_q_heads_ = 4,
        .num_kv_heads_ = 4, .head_dim_ = 64, .cache_block_size_ = 16});
    EXPECT_FALSE(r.has_value());
    EXPECT_EQ(r.error(), ErrorCode::InvalidArgument);
}

}  // namespace
}  // namespace engine
}  // namespace ccinfer
