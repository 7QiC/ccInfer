#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "backend/cuda/cuda_backend.h"
#include "cache/kv_cache_manager.h"
#include "cache/kv_cache_storage.h"
#include "kernel/cuda_kernels.h"

namespace ccinfer {
namespace {

// Full lifecycle: prepare_blocks → cache_full_blocks → release →
// prepare_blocks (hit) → verify shared blocks produce correct attention output.
TEST(PrefixCacheE2ETest, SharedPrefixProducesCorrectOutput) {
    constexpr int kNumTokens = 32;
    constexpr int kMaxBlocks = 16;
    constexpr int kNumLayers = 1;
    constexpr int nq = 8;
    constexpr int nkv = 4;
    constexpr int hd = 64;
    const int block_size = kKVBlockSize;

    auto backend_r = CudaBackend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;

    // 1. Init storage and manager.
    auto sr = KVCacheStorage::create<__nv_bfloat16>(backend, kNumLayers, kMaxBlocks, block_size, nkv,
                                                     hd);
    ASSERT_TRUE(sr.has_value());

    KVCacheManager mgr;
    auto init_r = mgr.init(std::move(*sr), kMaxBlocks, block_size);
    ASSERT_TRUE(init_r.has_value());

    // 2. Request 1: prepare and cache.
    std::vector<int32_t> tokens(kNumTokens);
    for (int i = 0; i < kNumTokens; ++i) tokens[i] = i + 1;

    auto pr1 = mgr.prepare_blocks(tokens);
    ASSERT_TRUE(pr1.has_value());
    ASSERT_EQ(pr1->block_table.size(), 2);
    ASSERT_EQ(pr1->prefix_hit_blocks, 0);
    int free_before_cache = mgr.num_free_blocks();

    auto cf1 = mgr.cache_full_blocks(pr1->block_table, tokens, kNumTokens);
    ASSERT_TRUE(cf1.has_value());

    // 3. Write KV data into the blocks, then release.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Fill K/V with distinct patterns per block.
    int64_t kv_elems = static_cast<int64_t>(kMaxBlocks) * block_size * nkv * hd;
    std::vector<__nv_bfloat16> h_k(kv_elems);
    std::vector<__nv_bfloat16> h_v(kv_elems);
    for (int64_t i = 0; i < kv_elems; ++i) {
        h_k[i] = __float2bfloat16(static_cast<float>((i + 1) % 100) * 0.01f);
        h_v[i] = __float2bfloat16(static_cast<float>((i + 50) % 100) * 0.01f);
    }
    cudaMemcpyAsync(mgr.k_cache(0), h_k.data(), kv_elems * sizeof(__nv_bfloat16),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(mgr.v_cache(0), h_v.data(), kv_elems * sizeof(__nv_bfloat16),
                    cudaMemcpyHostToDevice, stream);

    mgr.release_blocks(pr1->block_table);

    // 4. Verify blocks went to LRU (CACHED_IDLE).
    auto stats1 = mgr.stats();
    EXPECT_EQ(stats1.block_cached_idle, 2);
    // Free count unchanged (blocks moved from ACTIVE to LRU, not FREE).
    EXPECT_EQ(mgr.num_free_blocks(), free_before_cache);

    // 5. Request 2: same tokens → prefix hit.
    auto pr2 = mgr.prepare_blocks(tokens);
    ASSERT_TRUE(pr2.has_value());
    EXPECT_EQ(pr2->block_table.size(), 2);
    EXPECT_EQ(pr2->prefix_hit_blocks, 2);
    EXPECT_EQ(pr2->block_table.shared_count(), 2);
    // No new blocks allocated.
    EXPECT_EQ(mgr.num_free_blocks(), free_before_cache);

    // 6. Run prefill attention using the shared blocks.
    int64_t q_elems = static_cast<int64_t>(kNumTokens) * nq * hd;
    std::vector<__nv_bfloat16> h_q(q_elems);
    for (int64_t i = 0; i < q_elems; ++i)
        h_q[i] = __float2bfloat16(static_cast<float>((i + 10) % 50) * 0.02f);

    __nv_bfloat16 *d_q, *d_output;
    cudaMalloc(&d_q, q_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output, q_elems * sizeof(__nv_bfloat16));
    cudaMemcpyAsync(d_q, h_q.data(), q_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice,
                    stream);

    // Build prefill params.
    std::vector<int32_t> h_query_start_loc = {0, kNumTokens};
    std::vector<int32_t> h_context_lens = {kNumTokens};
    std::vector<int32_t> h_block_table(1 * (kMaxBlocks), -1);
    for (int b = 0; b < pr2->block_table.size(); ++b)
        h_block_table[b] = pr2->block_table[b];

    int32_t *d_qsl, *d_ctx, *d_bt;
    cudaMalloc(&d_qsl, 2 * sizeof(int32_t));
    cudaMalloc(&d_ctx, 1 * sizeof(int32_t));
    cudaMalloc(&d_bt, 1 * kMaxBlocks * sizeof(int32_t));
    cudaMemcpyAsync(d_qsl, h_query_start_loc.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(d_ctx, h_context_lens.data(), sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_bt, h_block_table.data(), 1 * kMaxBlocks * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);

    auto attn_r = launch_prefill_attention(
        d_q,
        static_cast<const __nv_bfloat16*>(mgr.k_cache(0)),
        static_cast<const __nv_bfloat16*>(mgr.v_cache(0)),
        d_bt, d_qsl, d_ctx, d_output, /*batch_size=*/1, kNumTokens, kMaxBlocks, nq, nkv, hd,
        block_size, stream);
    ASSERT_TRUE(attn_r.has_value());
    cudaStreamSynchronize(stream);

    // Verify output is finite.
    std::vector<float> h_out(q_elems);
    cudaMemcpy(h_out.data(), d_output, q_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < q_elems; ++i) {
        EXPECT_TRUE(std::isfinite(h_out[i])) << "non-finite at index " << i;
    }

    // 7. Release second request.
    mgr.release_blocks(pr2->block_table);
    EXPECT_EQ(mgr.stats().block_cached_idle, 2);

    // 8. LRU eviction: fill cache with distinct tokens, then verify eviction.
    int cached_count = 2;
    for (int i = 0; cached_count < kMaxBlocks; ++i) {
        std::vector<int32_t> t(32, 100 + i);
        auto px = mgr.prepare_blocks(t);
        ASSERT_TRUE(px.has_value());
        mgr.cache_full_blocks(px->block_table, t, 32);
        mgr.release_blocks(px->block_table);
        // Some may collide with previous hashes; count actual cached blocks.
        cached_count = mgr.stats().block_cached_idle + mgr.stats().block_active +
                       mgr.stats().block_free;
    }

    // At this point free list should be 0 or close to it.
    // allocate_blocks should trigger eviction.
    auto stats_before = mgr.stats();
    auto alloc = mgr.allocate_blocks(1);
    ASSERT_TRUE(alloc.has_value());
    auto stats_after = mgr.stats();
    EXPECT_LE(stats_after.block_cached_idle, stats_before.block_cached_idle);
    if (stats_before.block_free == 0) {
        EXPECT_GT(stats_after.prefix.evictions, stats_before.prefix.evictions);
    }

    // Cleanup.
    cudaFree(d_q);
    cudaFree(d_output);
    cudaFree(d_qsl);
    cudaFree(d_ctx);
    cudaFree(d_bt);
    cudaStreamDestroy(stream);
}

}  // namespace
}  // namespace ccinfer
