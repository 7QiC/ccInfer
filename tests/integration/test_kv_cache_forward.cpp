#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include "engine/backend/cuda/cuda_backend.h"
#include "engine/cache/block.h"
#include "engine/cache/kv_cache_manager.h"
#include "engine/cache/kv_cache_storage.h"
#include "engine/kernel/cuda_kernels.h"

namespace ccinfer {
namespace engine {
namespace {

// Full lifecycle test: alloc → write → prefill → decode → release
TEST(KVCacheE2ETest, PrefillAndDecodeWithRelease) {
    constexpr int kNumTokens = 32;
    constexpr int kDecodeBatch = 2;
    constexpr int kMaxBlocks = 16;
    constexpr int kNumLayers = 1;
    constexpr int nq = 8;
    constexpr int nkv = 4;
    constexpr int hd = 64;
    const int block_size = kKVBlockSize;

    // 1. Init GPU storage and CPU block manager separately
    CudaBackend backend;
    ASSERT_TRUE(backend.init(0).has_value());
    KVCacheStorage storage;
    auto r_storage = storage.init<__nv_bfloat16>(backend, kNumLayers, kMaxBlocks, block_size, nkv, hd);
    ASSERT_TRUE(r_storage.has_value());

    KVCacheManager mgr;
    auto init_r = mgr.init(kMaxBlocks);
    ASSERT_TRUE(init_r.has_value());
    EXPECT_EQ(mgr.num_free_blocks(), kMaxBlocks);

    // 2. allocate_blocks
    int num_blocks = (kNumTokens + block_size - 1) / block_size;
    auto alloc = mgr.allocate_blocks(num_blocks);
    ASSERT_TRUE(alloc.has_value());
    EXPECT_EQ(alloc->size(), num_blocks);
    EXPECT_EQ(mgr.num_free_blocks(), kMaxBlocks - num_blocks);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate Q, K, V in token-major dense layout
    int64_t q_elems = static_cast<int64_t>(kNumTokens) * nq * hd;
    int64_t kv_new_elems = static_cast<int64_t>(kNumTokens) * nkv * hd;
    int64_t cache_elems = static_cast<int64_t>(kMaxBlocks) * block_size * nkv * hd;

    __nv_bfloat16 *d_q, *d_k_new, *d_v_new, *d_out_prefill, *d_out_ref;
    cudaMalloc(&d_q, q_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_k_new, kv_new_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_v_new, kv_new_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out_prefill, q_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out_ref, q_elems * sizeof(__nv_bfloat16));

    // Random data
    std::vector<__nv_bfloat16> h(q_elems);
    for (int64_t i = 0; i < q_elems; ++i)
        h[i] = __float2bfloat16((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f);
    cudaMemcpy(d_q, h.data(), q_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    h.resize(kv_new_elems);
    for (int64_t i = 0; i < kv_new_elems; ++i)
        h[i] = __float2bfloat16((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f);
    cudaMemcpy(d_k_new, h.data(), kv_new_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    for (int64_t i = 0; i < kv_new_elems; ++i)
        h[i] = __float2bfloat16((static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f);
    cudaMemcpy(d_v_new, h.data(), kv_new_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Compute slot_mapping from allocated blocks
    std::vector<int32_t> slot_mapping(kNumTokens);
    for (int t = 0; t < kNumTokens; ++t) {
        int block_idx = t / block_size;
        int pos_in_block = t % block_size;
        slot_mapping[t] = (*alloc)[block_idx] * block_size + pos_in_block;
    }

    // Upload slot_mapping
    int32_t* d_slot_mapping;
    cudaMalloc(&d_slot_mapping, kNumTokens * sizeof(int32_t));
    cudaMemcpy(d_slot_mapping, slot_mapping.data(),
               kNumTokens * sizeof(int32_t), cudaMemcpyHostToDevice);

    // 3. write_kv_cache for each layer (use layer 0 for this test)
    {
        auto r = launch_write_kv_cache(d_k_new, d_v_new,
                                       static_cast<__nv_bfloat16*>(storage.k_layer(0)),
                                       static_cast<__nv_bfloat16*>(storage.v_layer(0)),
                                       d_slot_mapping, kNumTokens, nkv, hd,
                                       kMaxBlocks * block_size, stream);
        ASSERT_TRUE(r.has_value());
    }

    // 4. Upload block_table and metadata for prefill
    std::vector<int32_t> h_query_start_loc = {0, kNumTokens};
    std::vector<int32_t> h_context_lens = {kNumTokens};

    int32_t *d_query_start_loc, *d_context_lens, *d_block_table;
    cudaMalloc(&d_query_start_loc, 2 * sizeof(int32_t));
    cudaMalloc(&d_context_lens, 1 * sizeof(int32_t));
    cudaMalloc(&d_block_table, num_blocks * sizeof(int32_t));

    cudaMemcpyAsync(d_query_start_loc, h_query_start_loc.data(),
                    2 * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_context_lens, h_context_lens.data(),
                    sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_block_table, (*alloc).data(),
                    num_blocks * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    // 5. prefill_attention and compare with naive_attention
    {
        // Paged prefill: reads K/V through block_table
        auto r = launch_prefill_attention(d_q,
                                          static_cast<const __nv_bfloat16*>(storage.k_layer(0)),
                                          static_cast<const __nv_bfloat16*>(storage.v_layer(0)),
                                          d_block_table, d_query_start_loc, d_context_lens,
                                          d_out_prefill, 1, kNumTokens, num_blocks,
                                          nq, nkv, hd, block_size, stream);
        ASSERT_TRUE(r.has_value());

        // Reference: naive_attention on dense K/V
        r = launch_naive_attention(d_q, d_k_new, d_v_new, d_out_ref,
                                   kNumTokens, nq, nkv, hd, stream);
        ASSERT_TRUE(r.has_value());

        cudaStreamSynchronize(stream);

        // Compare bitwise
        std::vector<__nv_bfloat16> h_prefill(q_elems);
        std::vector<__nv_bfloat16> h_ref(q_elems);
        cudaMemcpy(h_prefill.data(), d_out_prefill, q_elems * sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ref.data(), d_out_ref, q_elems * sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToHost);

        float max_diff = 0.0f;
        for (int64_t i = 0; i < q_elems; ++i) {
            float diff = fabsf(__bfloat162float(h_prefill[i]) - __bfloat162float(h_ref[i]));
            if (diff > max_diff) max_diff = diff;
        }
        EXPECT_LT(max_diff, 0.02f) << "prefill paged vs naive max diff: " << max_diff;
    }

    // 6. decode_attention
    {
        // Two requests with different Q values
        int64_t decode_q_elems = static_cast<int64_t>(kDecodeBatch) * nq * hd;
        std::vector<__nv_bfloat16> h_decode_q(decode_q_elems);
        for (int64_t i = 0; i < decode_q_elems; ++i)
            h_decode_q[i] = __float2bfloat16(0.0f);
        // Request 0: Q = 0.1
        for (int h = 0; h < nq; ++h)
            for (int d = 0; d < hd; ++d)
                h_decode_q[h * hd + d] = __float2bfloat16(0.1f);
        // Request 1: Q = 0.3
        for (int h = 0; h < nq; ++h)
            for (int d = 0; d < hd; ++d)
                h_decode_q[nq * hd + h * hd + d] = __float2bfloat16(0.3f);

        __nv_bfloat16 *d_decode_q, *d_decode_out;
        cudaMalloc(&d_decode_q, decode_q_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&d_decode_out, decode_q_elems * sizeof(__nv_bfloat16));
        cudaMemcpy(d_decode_q, h_decode_q.data(),
                   decode_q_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

        // Two requests, both seeing the full 32-token context
        std::vector<int32_t> h_dec_block_table = {(*alloc)[0], (*alloc)[1],
                                                   (*alloc)[0], (*alloc)[1]};
        std::vector<int32_t> h_dec_context_lens = {kNumTokens, kNumTokens};

        int32_t *d_dec_block_table, *d_dec_context_lens;
        cudaMalloc(&d_dec_block_table, kDecodeBatch * num_blocks * sizeof(int32_t));
        cudaMalloc(&d_dec_context_lens, kDecodeBatch * sizeof(int32_t));
        cudaMemcpyAsync(d_dec_block_table, h_dec_block_table.data(),
                        kDecodeBatch * num_blocks * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_dec_context_lens, h_dec_context_lens.data(),
                        kDecodeBatch * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

        // Paged decode
        auto r = launch_decode_attention(d_decode_q,
                                          static_cast<const __nv_bfloat16*>(storage.k_layer(0)),
                                          static_cast<const __nv_bfloat16*>(storage.v_layer(0)),
                                         d_dec_block_table, d_dec_context_lens,
                                         d_decode_out, kDecodeBatch, num_blocks,
                                         nq, nkv, hd, block_size, stream);
        ASSERT_TRUE(r.has_value());
        cudaStreamSynchronize(stream);

        // Verify output is finite and per-request outputs differ from zero
        std::vector<__nv_bfloat16> h_decode_out(decode_q_elems);
        cudaMemcpy(h_decode_out.data(), d_decode_out,
                   decode_q_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

        for (int64_t i = 0; i < decode_q_elems; ++i) {
            EXPECT_TRUE(std::isfinite(__bfloat162float(h_decode_out[i])));
        }

        // With identical K/V context and different Q, outputs should differ
        float v0 = __bfloat162float(h_decode_out[0]);
        float v1 = __bfloat162float(h_decode_out[nq * hd]);
        EXPECT_NE(v0, v1) << "different Q should produce different outputs";

        cudaFree(d_decode_q); cudaFree(d_decode_out);
        cudaFree(d_dec_block_table); cudaFree(d_dec_context_lens);
    }

    // 7. release_blocks returns blocks to free list
    auto rel = mgr.release_blocks(*alloc);
    ASSERT_TRUE(rel.has_value());
    EXPECT_EQ(mgr.num_free_blocks(), kMaxBlocks);

    // Double release should fail
    auto rel2 = mgr.release_blocks(*alloc);
    EXPECT_FALSE(rel2.has_value());
    EXPECT_EQ(rel2.error(), ErrorCode::KVBlockDoubleFree);

    auto sync_err = cudaStreamSynchronize(stream);
    ASSERT_EQ(sync_err, cudaSuccess);
    auto last_err = cudaGetLastError();
    ASSERT_EQ(last_err, cudaSuccess);

    cudaStreamDestroy(stream);
    cudaFree(d_q); cudaFree(d_k_new); cudaFree(d_v_new);
    cudaFree(d_out_prefill); cudaFree(d_out_ref);
    cudaFree(d_slot_mapping);
    cudaFree(d_query_start_loc); cudaFree(d_context_lens);
    cudaFree(d_block_table);
}

}  // namespace
}  // namespace engine
}  // namespace ccinfer
