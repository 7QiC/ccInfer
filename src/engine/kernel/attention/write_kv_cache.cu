#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "engine/backend/cuda/cuda_utils.h"
#include "engine/kernel/cuda_kernels.h"

namespace ccinfer {
namespace engine {
namespace {

__global__ void write_kv_cache_kernel(const __nv_bfloat16* __restrict__ k_new,
                                      const __nv_bfloat16* __restrict__ v_new,
                                      __nv_bfloat16* __restrict__ k_cache,
                                      __nv_bfloat16* __restrict__ v_cache,
                                      const int32_t* __restrict__ slot_mapping,
                                      int n_kv_heads, int head_dim, int total_tokens,
                                      int max_slots) {

    int token_idx = blockIdx.x;
    int kv_head = blockIdx.y;
    int d = threadIdx.x;

    if (token_idx >= total_tokens || kv_head >= n_kv_heads || d >= head_dim) {
        return;
    }

    int slot = slot_mapping[token_idx];
    if (slot < 0 || slot >= max_slots) return;

    int64_t new_idx = (static_cast<int64_t>(token_idx) * n_kv_heads + kv_head) * head_dim + d;
    int64_t cache_idx = (static_cast<int64_t>(slot) * n_kv_heads + kv_head) * head_dim + d;

    k_cache[cache_idx] = k_new[new_idx];
    v_cache[cache_idx] = v_new[new_idx];
}

}  // namespace

Result<void> launch_write_kv_cache(const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
                                   __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                                   const int32_t* slot_mapping, int total_tokens,
                                   int n_kv_heads, int head_dim, int max_slots,
                                   cudaStream_t stream) {
    dim3 grid(total_tokens, n_kv_heads);
    dim3 block(head_dim);

    write_kv_cache_kernel<<<grid, block, 0, stream>>>(
        k_new, v_new, k_cache, v_cache, slot_mapping, n_kv_heads, head_dim, total_tokens,
        max_slots);

    return cuda_check(cudaGetLastError());
}

}  // namespace engine
}  // namespace ccinfer
