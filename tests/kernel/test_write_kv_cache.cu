#include <gtest/gtest.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

#include "backend/cuda/cuda_backend.h"
#include "cache/block.h"
#include "kernel/cuda_kernels.h"

namespace ccinfer {
namespace {

TEST(WriteKVCacheKernelTest, WriteAndReadbackNonContiguousSlots) {
    int total_tokens = 4;
    int n_kv_heads = 2;
    int head_dim = 4;
    int max_slots = 64;

    std::vector<__nv_bfloat16> h_k_new(total_tokens * n_kv_heads * head_dim);
    std::vector<__nv_bfloat16> h_v_new(total_tokens * n_kv_heads * head_dim);
    for (int t = 0; t < total_tokens; ++t) {
        for (int h = 0; h < n_kv_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                int64_t idx = (static_cast<int64_t>(t) * n_kv_heads + h) * head_dim + d;
                h_k_new[idx] = __float2bfloat16(static_cast<float>(t * 100 + h * 10 + d));
                h_v_new[idx] = __float2bfloat16(static_cast<float>(t * 200 + h * 10 + d));
            }
        }
    }

    std::vector<int32_t> h_slot_mapping = {5, 17, 0, 63};

    __nv_bfloat16 *d_k_new, *d_v_new, *d_k_cache, *d_v_cache;
    int32_t* d_slot_mapping;
    size_t kv_new_bytes = total_tokens * n_kv_heads * head_dim * sizeof(__nv_bfloat16);
    size_t kv_cache_bytes = max_slots * n_kv_heads * head_dim * sizeof(__nv_bfloat16);

    cudaMalloc(&d_k_new, kv_new_bytes);
    cudaMalloc(&d_v_new, kv_new_bytes);
    cudaMalloc(&d_k_cache, kv_cache_bytes);
    cudaMalloc(&d_v_cache, kv_cache_bytes);
    cudaMalloc(&d_slot_mapping, total_tokens * sizeof(int32_t));

    cudaMemcpy(d_k_new, h_k_new.data(), kv_new_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_new, h_v_new.data(), kv_new_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_slot_mapping, h_slot_mapping.data(),
               total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_k_cache, 0, kv_cache_bytes);
    cudaMemset(d_v_cache, 0, kv_cache_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto r = launch_write_kv_cache(d_k_new, d_v_new, d_k_cache, d_v_cache,
                                   d_slot_mapping, total_tokens, n_kv_heads,
                                   head_dim, max_slots, stream);
    ASSERT_TRUE(r.has_value());
    cudaStreamSynchronize(stream);

    for (int t = 0; t < total_tokens; ++t) {
        int slot = h_slot_mapping[t];
        std::vector<__nv_bfloat16> k_slot(n_kv_heads * head_dim);
        std::vector<__nv_bfloat16> v_slot(n_kv_heads * head_dim);

        cudaMemcpy(k_slot.data(),
                   d_k_cache + static_cast<int64_t>(slot) * n_kv_heads * head_dim,
                   n_kv_heads * head_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        cudaMemcpy(v_slot.data(),
                   d_v_cache + static_cast<int64_t>(slot) * n_kv_heads * head_dim,
                   n_kv_heads * head_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

        for (int h = 0; h < n_kv_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                EXPECT_EQ(__bfloat16_as_ushort(k_slot[h * head_dim + d]),
                          __bfloat16_as_ushort(h_k_new[(static_cast<int64_t>(t) * n_kv_heads + h) * head_dim + d]));
                EXPECT_EQ(__bfloat16_as_ushort(v_slot[h * head_dim + d]),
                          __bfloat16_as_ushort(h_v_new[(static_cast<int64_t>(t) * n_kv_heads + h) * head_dim + d]));
            }
        }
    }

    // Unwritten slot should stay zero
    std::vector<__nv_bfloat16> k_unused(n_kv_heads * head_dim);
    cudaMemcpy(k_unused.data(), d_k_cache + 1LL * n_kv_heads * head_dim,
               n_kv_heads * head_dim * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    EXPECT_FLOAT_EQ(__bfloat162float(k_unused[0]), 0.0f);

    auto sync_err = cudaStreamSynchronize(stream);
    ASSERT_EQ(sync_err, cudaSuccess);
    auto last_err = cudaGetLastError();
    ASSERT_EQ(last_err, cudaSuccess);
    cudaStreamDestroy(stream);
    cudaFree(d_k_new); cudaFree(d_v_new);
    cudaFree(d_k_cache); cudaFree(d_v_cache);
    cudaFree(d_slot_mapping);
}

TEST(WriteKVCacheKernelTest, NullPointerReturnsError) {
    auto backend_r = CudaBackend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;

    auto r = backend.template write_kv_cache<__nv_bfloat16>(WriteKVCacheParams{
        .k_new_ = nullptr, .v_new_ = nullptr, .k_cache_ = nullptr, .v_cache_ = nullptr,
        .slot_mapping_ = nullptr, .total_tokens_ = 1, .num_kv_heads_ = 1,
        .head_dim_ = 1, .max_slots_ = 64});
    EXPECT_FALSE(r.has_value());
    EXPECT_EQ(r.error(), ErrorCode::InvalidArgument);
}

}  // namespace
}  // namespace ccinfer
