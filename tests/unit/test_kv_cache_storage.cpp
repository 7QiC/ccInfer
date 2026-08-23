#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "backend/backend.h"
#include "cache/block.h"
#include "cache/kv_cache_storage.h"

namespace ccinfer {
namespace {

TEST(KVCacheStorageTest, LayerOffsetsAreCorrect) {
    constexpr int kNumLayers = 3;
    constexpr int kMaxBlocks = 2;
    constexpr int kNvKVHeads = 1;
    constexpr int kHeadDim = 4;

    auto backend_r = Backend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;
    auto s_r = KVCacheStorage::create(backend, kNumLayers, kMaxBlocks, kKVBlockSize, kNvKVHeads,
                                      kHeadDim, ccop::DType::kBFloat16);
    ASSERT_TRUE(s_r.has_value());
    auto& storage = **s_r;

    const int64_t layer_bytes = static_cast<int64_t>(kMaxBlocks) * kKVBlockSize * kNvKVHeads *
                                kHeadDim * sizeof(__nv_bfloat16);

    Tensor k0 = storage.k_layer_tensor(0);
    Tensor k1 = storage.k_layer_tensor(1);
    Tensor k2 = storage.k_layer_tensor(2);
    EXPECT_EQ(k0.data(), static_cast<char*>(k1.data()) - layer_bytes);
    EXPECT_EQ(k1.data(), static_cast<char*>(k2.data()) - layer_bytes);

    Tensor v0 = storage.v_layer_tensor(0);
    Tensor v1 = storage.v_layer_tensor(1);
    Tensor v2 = storage.v_layer_tensor(2);
    EXPECT_EQ(v0.data(), static_cast<char*>(v1.data()) - layer_bytes);
    EXPECT_EQ(v1.data(), static_cast<char*>(v2.data()) - layer_bytes);

    EXPECT_NE(k0.data(), v0.data());
}

TEST(KVCacheStorageTest, LayersAreIndependent) {
    constexpr int kNumLayers = 2;
    constexpr int kMaxBlocks = 1;
    constexpr int kNvKVHeads = 1;
    constexpr int kHeadDim = 4;

    auto backend_r = Backend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;
    auto s_r = KVCacheStorage::create(backend, kNumLayers, kMaxBlocks, kKVBlockSize, kNvKVHeads,
                                      kHeadDim, ccop::DType::kBFloat16);
    ASSERT_TRUE(s_r.has_value());
    auto& storage = **s_r;

    const int64_t layer_elems =
        static_cast<int64_t>(kMaxBlocks) * kKVBlockSize * kNvKVHeads * kHeadDim;
    const int64_t layer_bytes = layer_elems * sizeof(__nv_bfloat16);

    std::vector<__nv_bfloat16> data0(static_cast<size_t>(layer_elems));
    std::vector<__nv_bfloat16> data1(static_cast<size_t>(layer_elems));
    for (int64_t i = 0; i < layer_elems; ++i) {
        data0[static_cast<size_t>(i)] = __float2bfloat16(static_cast<float>(i));
        data1[static_cast<size_t>(i)] = __float2bfloat16(static_cast<float>(100 + i));
    }
    ASSERT_EQ(cudaMemcpy(storage.k_layer_tensor(0).data(), data0.data(), layer_bytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(storage.k_layer_tensor(1).data(), data1.data(), layer_bytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<__nv_bfloat16> actual(static_cast<size_t>(layer_elems));
    ASSERT_EQ(cudaMemcpy(actual.data(), storage.k_layer_tensor(0).data(), layer_bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int64_t i = 0; i < layer_elems; ++i) {
        EXPECT_FLOAT_EQ(__bfloat162float(actual[static_cast<size_t>(i)]), static_cast<float>(i));
    }
    ASSERT_EQ(cudaMemcpy(actual.data(), storage.k_layer_tensor(1).data(), layer_bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int64_t i = 0; i < layer_elems; ++i) {
        EXPECT_FLOAT_EQ(__bfloat162float(actual[static_cast<size_t>(i)]),
                        static_cast<float>(100 + i));
    }

    ASSERT_EQ(cudaMemcpy(storage.v_layer_tensor(0).data(), data0.data(), layer_bytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(storage.v_layer_tensor(1).data(), data1.data(), layer_bytes,
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual.data(), storage.v_layer_tensor(0).data(), layer_bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int64_t i = 0; i < layer_elems; ++i) {
        EXPECT_FLOAT_EQ(__bfloat162float(actual[static_cast<size_t>(i)]), static_cast<float>(i));
    }
}

TEST(KVCacheStorageTest, ZeroInitialized) {
    constexpr int kNumLayers = 1;
    constexpr int kMaxBlocks = 4;
    constexpr int kNvKVHeads = 2;
    constexpr int kHeadDim = 32;

    auto backend_r = Backend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;
    auto s_r = KVCacheStorage::create(backend, kNumLayers, kMaxBlocks, kKVBlockSize, kNvKVHeads,
                                      kHeadDim, ccop::DType::kBFloat16);
    ASSERT_TRUE(s_r.has_value());
    auto& storage = **s_r;

    const int64_t layer_elems =
        static_cast<int64_t>(kMaxBlocks) * kKVBlockSize * kNvKVHeads * kHeadDim;
    const int64_t layer_bytes = layer_elems * sizeof(__nv_bfloat16);

    std::vector<__nv_bfloat16> host(static_cast<size_t>(layer_elems), __float2bfloat16(-1.0f));
    ASSERT_EQ(cudaMemcpy(host.data(), storage.k_layer_tensor(0).data(), layer_bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int64_t i = 0; i < layer_elems; ++i) {
        EXPECT_FLOAT_EQ(__bfloat162float(host[static_cast<size_t>(i)]), 0.0f);
    }

    host.assign(static_cast<size_t>(layer_elems), __float2bfloat16(-1.0f));
    ASSERT_EQ(cudaMemcpy(host.data(), storage.v_layer_tensor(0).data(), layer_bytes,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    for (int64_t i = 0; i < layer_elems; ++i) {
        EXPECT_FLOAT_EQ(__bfloat162float(host[static_cast<size_t>(i)]), 0.0f);
    }
}

}  // namespace
}  // namespace ccinfer
