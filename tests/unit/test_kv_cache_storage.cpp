#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

#include "engine/cache/block.h"
#include "engine/cache/kv_cache_storage.h"

namespace ccinfer {
namespace engine {
namespace {

TEST(KVCacheStorageTest, InitSucceeds) {
    KVCacheStorage<__nv_bfloat16> storage;
    auto r = storage.init(4, 100, kKVBlockSize, 8, 128);
    ASSERT_TRUE(r.has_value());

    EXPECT_NE(storage.k_data(), nullptr);
    EXPECT_NE(storage.v_data(), nullptr);
    // Layer 3 should be accessible
    EXPECT_NE(storage.k_layer(3), nullptr);
    EXPECT_NE(storage.v_layer(3), nullptr);
}

TEST(KVCacheStorageTest, LayerPointersDiffer) {
    KVCacheStorage<__nv_bfloat16> storage;
    auto r = storage.init(2, 10, kKVBlockSize, 4, 64);
    ASSERT_TRUE(r.has_value());

    auto* k0 = storage.k_layer(0);
    auto* k1 = storage.k_layer(1);
    EXPECT_NE(k0, k1);

    // Two layers should have distinct storage regions
    EXPECT_GT(k1 - k0, 0);
    EXPECT_NE(storage.v_layer(0), storage.v_layer(1));
}

TEST(KVCacheStorageTest, ZeroInitialized) {
    KVCacheStorage<__nv_bfloat16> storage;
    auto r = storage.init(1, 4, kKVBlockSize, 2, 32);
    ASSERT_TRUE(r.has_value());

    int64_t layer_elems = static_cast<int64_t>(4) * kKVBlockSize * 2 * 32;
    std::vector<float> host_k(layer_elems, -1.0f);
    cudaMemcpy(host_k.data(), storage.k_layer(0),
               layer_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_FLOAT_EQ(host_k[0], 0.0f);
}

}  // namespace
}  // namespace engine
}  // namespace ccinfer
