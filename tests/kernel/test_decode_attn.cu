#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "backend/cuda/cuda_backend.h"
#include "cache/block.h"
#include "kernel/cuda_kernels.h"

namespace ccinfer {
namespace {

void fill_bf16(__nv_bfloat16* d, const std::vector<float>& h, cudaStream_t stream) {
    std::vector<__nv_bfloat16> buf(h.size());
    for (size_t i = 0; i < h.size(); ++i) buf[i] = __float2bfloat16(h[i]);
    cudaMemcpyAsync(d, buf.data(), buf.size() * sizeof(__nv_bfloat16),
                    cudaMemcpyHostToDevice, stream);
}

std::vector<float> decode_reference(const std::vector<__nv_bfloat16>& q,
                                    const std::vector<__nv_bfloat16>& k_cache,
                                    const std::vector<__nv_bfloat16>& v_cache,
                                    const std::vector<int32_t>& block_table,
                                    const std::vector<int32_t>& context_lens, int batch,
                                    int max_blocks_per_req, int num_q_heads, int num_kv_heads,
                                    int head_dim, int block_size) {
    std::vector<float> out(static_cast<size_t>(batch) * num_q_heads * head_dim, 0.0f);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int gqa = num_q_heads / num_kv_heads;

    for (int req = 0; req < batch; ++req) {
        const int context_len = context_lens[static_cast<size_t>(req)];
        for (int qh = 0; qh < num_q_heads; ++qh) {
            const int kvh = qh / gqa;
            std::vector<float> scores(static_cast<size_t>(context_len));
            float max_score = -INFINITY;

            for (int pos = 0; pos < context_len; ++pos) {
                const int logical_block = pos / block_size;
                const int offset = pos - logical_block * block_size;
                const int physical_block =
                    block_table[static_cast<size_t>(req) * max_blocks_per_req + logical_block];
                const int64_t slot = static_cast<int64_t>(physical_block) * block_size + offset;
                const int64_t k_base = (slot * num_kv_heads + kvh) * head_dim;
                const int64_t q_base = (static_cast<int64_t>(req) * num_q_heads + qh) * head_dim;

                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += __bfloat162float(q[static_cast<size_t>(q_base + d)]) *
                             __bfloat162float(k_cache[static_cast<size_t>(k_base + d)]);
                }
                score *= scale;
                scores[static_cast<size_t>(pos)] = score;
                max_score = std::max(max_score, score);
            }

            float denom = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - max_score);
                denom += score;
            }

            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int pos = 0; pos < context_len; ++pos) {
                    const int logical_block = pos / block_size;
                    const int offset = pos - logical_block * block_size;
                    const int physical_block =
                        block_table[static_cast<size_t>(req) * max_blocks_per_req + logical_block];
                    const int64_t slot = static_cast<int64_t>(physical_block) * block_size + offset;
                    const int64_t v_base = (slot * num_kv_heads + kvh) * head_dim;
                    acc += (scores[static_cast<size_t>(pos)] / denom) *
                           __bfloat162float(v_cache[static_cast<size_t>(v_base + d)]);
                }
                const int64_t out_idx = (static_cast<int64_t>(req) * num_q_heads + qh) * head_dim + d;
                out[static_cast<size_t>(out_idx)] = acc;
            }
        }
    }

    return out;
}

TEST(DecodeAttnKernelTest, SingleDecodeStep) {
    constexpr int kBatch = 2;
    constexpr int kNumQHeads = 4;
    constexpr int kNumKVHeads = 4;
    constexpr int kHeadDim = 64;
    constexpr int kContextLen = 32;
    constexpr int kNumBlocks = 2;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int64_t q_elems = static_cast<int64_t>(kBatch) * kNumQHeads * kHeadDim;
    int64_t kv_elems = static_cast<int64_t>(kNumBlocks) * kKVBlockSize * kNumKVHeads * kHeadDim;

    __nv_bfloat16 *d_q, *d_k_cache, *d_v_cache, *d_out;
    cudaMalloc(&d_q, q_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_k_cache, kv_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_v_cache, kv_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out, q_elems * sizeof(__nv_bfloat16));

    // Simple patterns: Q=0.1, K=0.1, V=0.2
    // Output should be uniform softmax → ~0.2 per dim
    std::vector<float> h_q(q_elems, 0.1f);
    std::vector<float> h_k(kv_elems);
    std::vector<float> h_v(kv_elems);
    for (int64_t i = 0; i < kv_elems; ++i) {
        h_k[i] = 0.1f;
        h_v[i] = 0.2f;
    }
    fill_bf16(d_q, h_q, stream);
    fill_bf16(d_k_cache, h_k, stream);
    fill_bf16(d_v_cache, h_v, stream);

    std::vector<int32_t> h_context_lens = {kContextLen, kContextLen};
    // One block_table row per request: [req0_block0, req0_block1, req1_block0, req1_block1]
    std::vector<int32_t> h_block_table = {0, 1, 0, 1};
    static_assert(kNumBlocks == 2);

    int32_t *d_context_lens, *d_block_table;
    cudaMalloc(&d_context_lens, kBatch * sizeof(int32_t));
    cudaMalloc(&d_block_table, kBatch * kNumBlocks * sizeof(int32_t));

    cudaMemcpyAsync(d_context_lens, h_context_lens.data(),
                    kBatch * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_block_table, h_block_table.data(),
                    kBatch * kNumBlocks * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    auto r = launch_decode_attention(d_q, d_k_cache, d_v_cache, d_block_table,
                                     d_context_lens, d_out, kBatch, kNumBlocks,
                                     kNumQHeads, kNumKVHeads, kHeadDim,
                                     kKVBlockSize, stream);
    ASSERT_TRUE(r.has_value());
    cudaStreamSynchronize(stream);

    // Verify output is finite
    std::vector<__nv_bfloat16> h_out(q_elems);
    cudaMemcpy(h_out.data(), d_out, q_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    for (int64_t i = 0; i < q_elems; ++i) {
        EXPECT_TRUE(std::isfinite(__bfloat162float(h_out[i])));
    }

    // Two different requests should produce different outputs if Q differs
    // With identical Q and KV, they should be the same
    float v0 = __bfloat162float(h_out[0]);
    float v1 = __bfloat162float(h_out[kNumQHeads * kHeadDim]);
    EXPECT_NEAR(v0, v1, 1e-3f);

    cudaFree(d_q); cudaFree(d_k_cache); cudaFree(d_v_cache); cudaFree(d_out);
    cudaFree(d_context_lens); cudaFree(d_block_table);
    auto sync_err = cudaStreamSynchronize(stream);
    ASSERT_EQ(sync_err, cudaSuccess);
    auto last_err = cudaGetLastError();
    ASSERT_EQ(last_err, cudaSuccess);
    cudaStreamDestroy(stream);
}

TEST(DecodeAttnKernelTest, GQA) {
    constexpr int kBatch = 1;
    constexpr int kNumQHeads = 8;
    constexpr int kNumKVHeads = 2;
    constexpr int kHeadDim = 64;
    constexpr int kContextLen = 16;
    constexpr int kNumBlocks = 1;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int64_t q_elems = static_cast<int64_t>(kBatch) * kNumQHeads * kHeadDim;
    int64_t kv_elems = static_cast<int64_t>(kNumBlocks) * kKVBlockSize * kNumKVHeads * kHeadDim;

    __nv_bfloat16 *d_q, *d_k_cache, *d_v_cache, *d_out;
    cudaMalloc(&d_q, q_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_k_cache, kv_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_v_cache, kv_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out, q_elems * sizeof(__nv_bfloat16));

    std::vector<float> h_q(q_elems, 0.1f);
    std::vector<float> h_kv(kv_elems, 0.1f);
    fill_bf16(d_q, h_q, stream);
    fill_bf16(d_k_cache, h_kv, stream);
    fill_bf16(d_v_cache, h_kv, stream);

    std::vector<int32_t> h_context_lens = {kContextLen};
    std::vector<int32_t> h_block_table = {0};
    int32_t *d_context_lens, *d_block_table;
    cudaMalloc(&d_context_lens, sizeof(int32_t));
    cudaMalloc(&d_block_table, sizeof(int32_t));
    cudaMemcpyAsync(d_context_lens, h_context_lens.data(), sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_block_table, h_block_table.data(), sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);

    auto r = launch_decode_attention(d_q, d_k_cache, d_v_cache, d_block_table,
                                     d_context_lens, d_out, kBatch, kNumBlocks,
                                     kNumQHeads, kNumKVHeads, kHeadDim,
                                     kKVBlockSize, stream);
    ASSERT_TRUE(r.has_value());
    cudaStreamSynchronize(stream);

    std::vector<__nv_bfloat16> h_out(q_elems);
    cudaMemcpy(h_out.data(), d_out, q_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Heads with same kv_head should produce same output (all inputs equal)
    for (int h = 1; h < kNumQHeads; ++h) {
        // Compare head h's first element with head 0's first element
        int64_t off0 = 0;
        int64_t off_h = static_cast<int64_t>(h) * kHeadDim;
        EXPECT_NEAR(__bfloat162float(h_out[off_h]),
                    __bfloat162float(h_out[off0]), 1e-2f);
    }

    cudaFree(d_q); cudaFree(d_k_cache); cudaFree(d_v_cache); cudaFree(d_out);
    cudaFree(d_context_lens); cudaFree(d_block_table);
    auto sync_err = cudaStreamSynchronize(stream);
    ASSERT_EQ(sync_err, cudaSuccess);
    auto last_err = cudaGetLastError();
    ASSERT_EQ(last_err, cudaSuccess);
    cudaStreamDestroy(stream);
}

TEST(DecodeAttnKernelTest, RandomPagedGQAMatchesCpuReference) {
    constexpr int kBatch = 2;
    constexpr int kNumQHeads = 8;
    constexpr int kNumKVHeads = 2;
    constexpr int kHeadDim = 64;
    constexpr int kMaxBlocksPerReq = 3;
    constexpr int kNumPhysicalBlocks = 3;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int64_t q_elems = static_cast<int64_t>(kBatch) * kNumQHeads * kHeadDim;
    const int64_t kv_elems =
        static_cast<int64_t>(kNumPhysicalBlocks) * kKVBlockSize * kNumKVHeads * kHeadDim;

    std::vector<__nv_bfloat16> h_q(static_cast<size_t>(q_elems));
    std::vector<__nv_bfloat16> h_k(static_cast<size_t>(kv_elems));
    std::vector<__nv_bfloat16> h_v(static_cast<size_t>(kv_elems));
    for (int64_t i = 0; i < q_elems; ++i) {
        float x = static_cast<float>((i * 17) % 23 - 11) * 0.03125f;
        h_q[static_cast<size_t>(i)] = __float2bfloat16(x);
    }
    for (int64_t i = 0; i < kv_elems; ++i) {
        float k = static_cast<float>((i * 13) % 29 - 14) * 0.025f;
        float v = static_cast<float>((i * 7) % 31 - 15) * 0.02f;
        h_k[static_cast<size_t>(i)] = __float2bfloat16(k);
        h_v[static_cast<size_t>(i)] = __float2bfloat16(v);
    }

    std::vector<int32_t> h_context_lens = {31, 17};
    std::vector<int32_t> h_block_table = {
        2, 0, 1,
        1, 2, -1,
    };
    auto expected = decode_reference(h_q, h_k, h_v, h_block_table, h_context_lens, kBatch,
                                     kMaxBlocksPerReq, kNumQHeads, kNumKVHeads, kHeadDim,
                                     kKVBlockSize);

    __nv_bfloat16 *d_q, *d_k_cache, *d_v_cache, *d_out;
    int32_t *d_context_lens, *d_block_table;
    cudaMalloc(&d_q, q_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_k_cache, kv_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_v_cache, kv_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out, q_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_context_lens, kBatch * sizeof(int32_t));
    cudaMalloc(&d_block_table, kBatch * kMaxBlocksPerReq * sizeof(int32_t));

    cudaMemcpyAsync(d_q, h_q.data(), q_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(d_k_cache, h_k.data(), kv_elems * sizeof(__nv_bfloat16),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_v_cache, h_v.data(), kv_elems * sizeof(__nv_bfloat16),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_context_lens, h_context_lens.data(), kBatch * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_block_table, h_block_table.data(),
                    kBatch * kMaxBlocksPerReq * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    auto r = launch_decode_attention(d_q, d_k_cache, d_v_cache, d_block_table, d_context_lens,
                                     d_out, kBatch, kMaxBlocksPerReq, kNumQHeads, kNumKVHeads,
                                     kHeadDim, kKVBlockSize, stream);
    ASSERT_TRUE(r.has_value());
    cudaStreamSynchronize(stream);

    std::vector<__nv_bfloat16> h_out(static_cast<size_t>(q_elems));
    cudaMemcpy(h_out.data(), d_out, q_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (int64_t i = 0; i < q_elems; ++i) {
        const float got = __bfloat162float(h_out[static_cast<size_t>(i)]);
        const float diff = std::abs(got - expected[static_cast<size_t>(i)]);
        max_diff = std::max(max_diff, diff);
    }
    EXPECT_LT(max_diff, 0.01f);

    cudaFree(d_q); cudaFree(d_k_cache); cudaFree(d_v_cache); cudaFree(d_out);
    cudaFree(d_context_lens); cudaFree(d_block_table);
    auto sync_err = cudaStreamSynchronize(stream);
    ASSERT_EQ(sync_err, cudaSuccess);
    auto last_err = cudaGetLastError();
    ASSERT_EQ(last_err, cudaSuccess);
    cudaStreamDestroy(stream);
}

TEST(DecodeAttnKernelTest, NullPointers) {
    auto backend_r = CudaBackend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;

    auto r = backend.template decode_attention<__nv_bfloat16>(DecodeAttnParams{
        .q_ = nullptr, .k_cache_ = nullptr, .v_cache_ = nullptr, .block_table_ = nullptr,
        .context_lens_ = nullptr, .output_ = nullptr,
        .batch_size_ = 1, .max_blocks_per_req_ = 1, .num_q_heads_ = 4,
        .num_kv_heads_ = 4, .head_dim_ = 64, .cache_block_size_ = 16});
    EXPECT_FALSE(r.has_value());
    EXPECT_EQ(r.error(), ErrorCode::InvalidArgument);
}

}  // namespace
}  // namespace ccinfer
