#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

#include "engine/backend/cuda/cuda_backend.h"
#include "engine/cache/block.h"
#include "engine/cache/kv_cache_storage.h"

namespace ccinfer {
namespace engine {
namespace {

TEST(KVCacheStorageTest, LayerOffsetsAreCorrect) {
    constexpr int kNumLayers = 3;
    constexpr int kMaxBlocks = 2;
    constexpr int kNvKVHeads = 1;
    constexpr int kHeadDim = 4;

    auto backend_r = CudaBackend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;
    KVCacheStorage storage;
    auto r = storage.init<__nv_bfloat16>(backend, kNumLayers, kMaxBlocks, kKVBlockSize,
                                         kNvKVHeads, kHeadDim);
    ASSERT_TRUE(r.has_value());

    int64_t layer_bytes =
        static_cast<int64_t>(kMaxBlocks) * kKVBlockSize * kNvKVHeads * kHeadDim *
        sizeof(__nv_bfloat16);

    // k_layer(0) == k_data() (first layer starts at beginning of buffer)
    EXPECT_EQ(storage.k_layer(0), storage.k_data());
    EXPECT_EQ(storage.v_layer(0), storage.v_data());

    // Successive k_layers should be exactly one layer's bytes apart
    auto* k0 = static_cast<char*>(storage.k_layer(0));
    auto* k1 = static_cast<char*>(storage.k_layer(1));
    auto* k2 = static_cast<char*>(storage.k_layer(2));
    EXPECT_EQ(k1 - k0, layer_bytes);
    EXPECT_EQ(k2 - k1, layer_bytes);

    // Same for v_layers
    auto* v0 = static_cast<char*>(storage.v_layer(0));
    auto* v1 = static_cast<char*>(storage.v_layer(1));
    auto* v2 = static_cast<char*>(storage.v_layer(2));
    EXPECT_EQ(v1 - v0, layer_bytes);
    EXPECT_EQ(v2 - v1, layer_bytes);

    // Distinct storage: k and v are separate allocations
    EXPECT_NE(storage.k_data(), storage.v_data());
    EXPECT_NE(k0, v0);
}

TEST(KVCacheStorageTest, LayersAreIndependent) {
    constexpr int kNumLayers = 2;
    constexpr int kMaxBlocks = 1;
    constexpr int kNvKVHeads = 1;
    constexpr int kHeadDim = 4;

    auto backend_r = CudaBackend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;
    KVCacheStorage storage;
    auto r = storage.init<__nv_bfloat16>(backend, kNumLayers, kMaxBlocks, kKVBlockSize,
                                         kNvKVHeads, kHeadDim);
    ASSERT_TRUE(r.has_value());

    int64_t layer_elems =
        static_cast<int64_t>(kMaxBlocks) * kKVBlockSize * kNvKVHeads * kHeadDim;
    int64_t layer_bytes = layer_elems * sizeof(__nv_bfloat16);

    // Write per-element patterns: layer 0 = {0.0, 1.0, 2.0, ...}, layer 1 = {100.0, 101.0, ...}
    std::vector<__nv_bfloat16> data0(layer_elems);
    std::vector<__nv_bfloat16> data1(layer_elems);
    for (int64_t i = 0; i < layer_elems; ++i) {
        data0[i] = __float2bfloat16(static_cast<float>(i));
        data1[i] = __float2bfloat16(static_cast<float>(100 + i));
    }
    ASSERT_EQ(cudaMemcpy(storage.k_layer(0), data0.data(), layer_bytes,
                         cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(storage.k_layer(1), data1.data(), layer_bytes,
                         cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Read back layer 0 — should be {0, 1, 2, ...}, not affected by layer 1 write
    std::vector<__nv_bfloat16> actual(layer_elems);
    ASSERT_EQ(cudaMemcpy(actual.data(), storage.k_layer(0), layer_bytes,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int64_t i = 0; i < layer_elems; ++i) {
        EXPECT_FLOAT_EQ(__bfloat162float(actual[i]), static_cast<float>(i))
            << "k_layer(0)[" << i << "] corrupted by k_layer(1) write";
    }

    // Read back layer 1 — should be {100, 101, 102, ...}
    ASSERT_EQ(cudaMemcpy(actual.data(), storage.k_layer(1), layer_bytes,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int64_t i = 0; i < layer_elems; ++i) {
        EXPECT_FLOAT_EQ(__bfloat162float(actual[i]), static_cast<float>(100 + i))
            << "k_layer(1)[" << i << "] incorrect";
    }

    // Same for v_layers
    ASSERT_EQ(cudaMemcpy(storage.v_layer(0), data0.data(), layer_bytes,
                         cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(storage.v_layer(1), data1.data(), layer_bytes,
                         cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(actual.data(), storage.v_layer(0), layer_bytes,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int64_t i = 0; i < layer_elems; ++i) {
        EXPECT_FLOAT_EQ(__bfloat162float(actual[i]), static_cast<float>(i))
            << "v_layer(0)[" << i << "] corrupted";
    }
}

TEST(KVCacheStorageTest, ZeroInitialized) {
    constexpr int kNumLayers = 1;
    constexpr int kMaxBlocks = 4;
    constexpr int kNvKVHeads = 2;
    constexpr int kHeadDim = 32;

    auto backend_r = CudaBackend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;
    KVCacheStorage storage;
    auto r = storage.init<__nv_bfloat16>(backend, kNumLayers, kMaxBlocks, kKVBlockSize,
                                         kNvKVHeads, kHeadDim);
    ASSERT_TRUE(r.has_value());

    int64_t layer_elems =
        static_cast<int64_t>(kMaxBlocks) * kKVBlockSize * kNvKVHeads * kHeadDim;
    int64_t layer_bytes = layer_elems * sizeof(__nv_bfloat16);

    // Read back k_layer(0) — should be all zeros from cudaMalloc
    std::vector<__nv_bfloat16> host(layer_elems, __float2bfloat16(-1.0f));
    ASSERT_EQ(cudaMemcpy(host.data(), storage.k_layer(0), layer_bytes,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int64_t i = 0; i < layer_elems; ++i) {
        EXPECT_FLOAT_EQ(__bfloat162float(host[i]), 0.0f)
            << "k_layer(0)[" << i << "] not zero-initialised";
    }

    // v_layer(0) should also be zero
    host.assign(layer_elems, __float2bfloat16(-1.0f));
    ASSERT_EQ(cudaMemcpy(host.data(), storage.v_layer(0), layer_bytes,
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int64_t i = 0; i < std::min(int64_t{16}, layer_elems); ++i) {
        EXPECT_FLOAT_EQ(__bfloat162float(host[i]), 0.0f)
            << "v_layer(0)[" << i << "] not zero-initialised";
    }
}

}  // namespace
}  // namespace engine
}  // namespace ccinfer
