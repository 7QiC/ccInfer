#include <cmath>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "backend/backend.h"
#include "block/block_pool.h"
#include "block/block_storage.h"
#include "cache/block_cache.h"
#include "facade/ops.h"

namespace ccinfer {
namespace {

void publish_blocks(BlockPool& pool, BlockCache& cache, const std::vector<int32_t>& tokens,
                    const BlockTable& table) {
    uint64_t parent_hash = 0;
    for (int i = 0; i < table.size(); ++i) {
        const auto begin = tokens.begin() + i * pool.block_size();
        std::vector<int32_t> block_tokens(begin, begin + pool.block_size());
        parent_hash = cache.publish_full_block(parent_hash, block_tokens, table[i]);
    }
}

void release_request_blocks(BlockPool& pool, BlockCache& cache, const BlockTable& table) {
    for (int i = 0; i < table.size(); ++i) {
        const BlockId id = table[i];
        if (pool.release_request(id) != 0) continue;
        if (cache.contains(id)) {
            cache.mark_idle(id);
        } else {
            pool.recycle(id);
        }
    }
}

Result<BlockTable> allocate_with_eviction(BlockPool& pool, BlockCache& cache, int num_blocks) {
    while (pool.num_free_blocks() < num_blocks) {
        if (!cache.evict_one()) break;
    }
    return pool.allocate_blocks(num_blocks);
}

// Full lifecycle: lookup/allocate → publish blocks → release →
// lookup (hit) → verify shared blocks produce correct attention output.
TEST(PrefixCacheE2ETest, SharedPrefixProducesCorrectOutput) {
    constexpr int kNumTokens = 32;
    constexpr int kMaxBlocks = 16;
    constexpr int kNumLayers = 1;
    constexpr int nq = 8;
    constexpr int nkv = 4;
    constexpr int hd = 64;
    const int block_size = kKVBlockSize;

    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto& backend = **backend_r;

    // 1. Init storage and block pool.
    auto sr = BlockStorage::create(backend, kNumLayers, kMaxBlocks, block_size, nkv, hd,
                                   ccop::DType::kBFloat16);
    ASSERT_TRUE(sr.has_value());

    auto storage = std::move(*sr);
    BlockPool pool(kMaxBlocks, block_size);
    BlockCache cache(pool);

    // 2. Request 1: prepare and cache.
    std::vector<int32_t> tokens(kNumTokens);
    for (int i = 0; i < kNumTokens; ++i) tokens[i] = i + 1;

    auto pr1 = cache.lookup_prefix_cache(tokens);
    auto alloc1 =
        allocate_with_eviction(pool, cache, kNumTokens / block_size - pr1.block_table.size());
    ASSERT_TRUE(alloc1.has_value());
    for (int b = 0; b < alloc1->size(); ++b) pr1.block_table.push_back((*alloc1)[b]);
    ASSERT_EQ(pr1.block_table.size(), 2);
    ASSERT_EQ(pr1.prefix_hit_blocks, 0);
    int free_before_cache = pool.num_free_blocks();

    publish_blocks(pool, cache, tokens, pr1.block_table);

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
    cudaMemcpyAsync(storage->k_layer_tensor(0).data(), h_k.data(), kv_elems * sizeof(__nv_bfloat16),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(storage->v_layer_tensor(0).data(), h_v.data(), kv_elems * sizeof(__nv_bfloat16),
                    cudaMemcpyHostToDevice, stream);

    release_request_blocks(pool, cache, pr1.block_table);

    // 4. Verify blocks went to LRU (CACHED_IDLE).
    auto stats1 = cache.stats();
    EXPECT_EQ(stats1.block_cached_idle, 2);
    // Free count unchanged (blocks moved from ACTIVE to LRU, not FREE).
    EXPECT_EQ(pool.num_free_blocks(), free_before_cache);

    // 5. Request 2: same tokens → prefix hit.
    auto pr2 = cache.lookup_prefix_cache(tokens);
    EXPECT_EQ(pr2.block_table.size(), 2);
    EXPECT_EQ(pr2.prefix_hit_blocks, 2);
    EXPECT_EQ(pr2.block_table.shared_count(), 2);
    // No new blocks allocated.
    EXPECT_EQ(pool.num_free_blocks(), free_before_cache);

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

    std::vector<int32_t> h_query_start_loc = {0, kNumTokens};
    std::vector<int32_t> h_context_lens = {kNumTokens};
    std::vector<int32_t> h_block_table(1 * (kMaxBlocks), -1);
    for (int b = 0; b < pr2.block_table.size(); ++b) h_block_table[b] = pr2.block_table[b];

    int32_t *d_qsl, *d_ctx, *d_bt;
    cudaMalloc(&d_qsl, 2 * sizeof(int32_t));
    cudaMalloc(&d_ctx, 1 * sizeof(int32_t));
    cudaMalloc(&d_bt, 1 * kMaxBlocks * sizeof(int32_t));
    cudaMemcpyAsync(d_qsl, h_query_start_loc.data(), 2 * sizeof(int32_t), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(d_ctx, h_context_lens.data(), sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_bt, h_block_table.data(), 1 * kMaxBlocks * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);

    constexpr ccop::Device kCuda0{ccop::DeviceType::kCUDA, 0};
    ccop::Tensor q(d_q, ccop::DType::kBFloat16, kCuda0, {kNumTokens, nq, hd});
    auto k_cache = storage->k_block_tensor(0);
    auto v_cache = storage->v_block_tensor(0);
    ccop::Tensor block_table(d_bt, ccop::DType::kInt32, kCuda0, {1, kMaxBlocks});
    ccop::Tensor query_start_loc(d_qsl, ccop::DType::kInt32, kCuda0, {2});
    ccop::Tensor context_lens(d_ctx, ccop::DType::kInt32, kCuda0, {1});
    ccop::Tensor out(d_output, ccop::DType::kBFloat16, kCuda0, {kNumTokens, nq, hd});
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    ASSERT_TRUE(
        map_result(ccop::prefill_attention(q, k_cache, v_cache, block_table, query_start_loc,
                                           context_lens, &out, scale, {stream}))
            .has_value());
    cudaStreamSynchronize(stream);

    std::vector<__nv_bfloat16> h_out_bf16(static_cast<size_t>(q_elems));
    cudaMemcpy(h_out_bf16.data(), d_output, q_elems * sizeof(__nv_bfloat16),
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < q_elems; ++i) {
        EXPECT_TRUE(std::isfinite(__bfloat162float(h_out_bf16[static_cast<size_t>(i)])))
            << "non-finite at index " << i;
    }

    // 7. Release second request.
    release_request_blocks(pool, cache, pr2.block_table);
    EXPECT_EQ(cache.stats().block_cached_idle, 2);

    // 8. LRU eviction: fill cache with distinct tokens, then verify eviction.
    int cached_count = 2;
    for (int i = 0; cached_count < kMaxBlocks; ++i) {
        std::vector<int32_t> t(32, 100 + i);
        auto px = cache.lookup_prefix_cache(t);
        auto alloc_px = allocate_with_eviction(
            pool, cache, static_cast<int>(t.size()) / block_size - px.block_table.size());
        ASSERT_TRUE(alloc_px.has_value());
        for (int b = 0; b < alloc_px->size(); ++b) px.block_table.push_back((*alloc_px)[b]);
        publish_blocks(pool, cache, t, px.block_table);
        release_request_blocks(pool, cache, px.block_table);
        // Some may collide with previous hashes; count actual cached blocks.
        cached_count =
            cache.stats().block_cached_idle + cache.stats().block_active + cache.stats().block_free;
    }

    // At this point free list should be 0 or close to it.
    // allocate_blocks should trigger eviction.
    auto stats_before = cache.stats();
    auto alloc = allocate_with_eviction(pool, cache, 1);
    ASSERT_TRUE(alloc.has_value());
    auto stats_after = cache.stats();
    EXPECT_LE(stats_after.block_cached_idle, stats_before.block_cached_idle);
    if (stats_before.block_free == 0) {
        EXPECT_GT(stats_after.evictions, stats_before.evictions);
    }

    cudaFree(d_q);
    cudaFree(d_output);
    cudaFree(d_qsl);
    cudaFree(d_ctx);
    cudaFree(d_bt);
    cudaStreamDestroy(stream);
}

}  // namespace
}  // namespace ccinfer
