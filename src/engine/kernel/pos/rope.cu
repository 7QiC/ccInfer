#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "engine/backend/cuda/cuda_utils.h"
#include "engine/kernel/common.h"
#include "engine/kernel/cuda_kernels.h"

namespace ccinfer {
namespace engine {

namespace {

// Scalar split-half RoPE with bf16
__global__ void rope_split_half_scalar_kernel(__nv_bfloat16* __restrict__ q,
                                              __nv_bfloat16* __restrict__ k,
                                              const int32_t* __restrict__ positions,
                                              const float2* __restrict__ rope_cache, int num_tokens,
                                              int num_q_heads, int num_kv_heads, int head_dim,
                                              int rotary_dim, int rope_cache_max_position) {
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    const int logical_head = blockIdx.z;

    const int half_rotary_dim = rotary_dim / 2;
    const int total_heads = num_q_heads + num_kv_heads;
    if (token >= num_tokens || logical_head >= total_heads || pair_idx >= half_rotary_dim) return;

    const bool is_q = logical_head < num_q_heads;
    const int head = is_q ? logical_head : logical_head - num_q_heads;
    const int num_heads = is_q ? num_q_heads : num_kv_heads;
    __nv_bfloat16* x = is_q ? q : k;

    const int pos = positions[token];
    if (pos < 0 || pos >= rope_cache_max_position) return;

    const int64_t base = (static_cast<int64_t>(token) * num_heads + head) * head_dim;
    const int i0 = pair_idx;
    const int i1 = pair_idx + half_rotary_dim;

    const float2 cs = rope_cache[static_cast<int64_t>(pos) * half_rotary_dim + pair_idx];
    const float x0 = __bfloat162float(x[base + i0]);
    const float x1 = __bfloat162float(x[base + i1]);

    x[base + i0] = __float2bfloat16_rn(x0 * cs.x - x1 * cs.y);
    x[base + i1] = __float2bfloat16_rn(x1 * cs.x + x0 * cs.y);
}

// BF16 half2 fast path: two pairs per thread
__global__ void rope_split_half_bf162_kernel(__nv_bfloat16* __restrict__ q,
                                             __nv_bfloat16* __restrict__ k,
                                             const int32_t* __restrict__ positions,
                                             const float2* __restrict__ rope_cache, int num_tokens,
                                             int num_q_heads, int num_kv_heads, int head_dim,
                                             int rotary_dim, int rope_cache_max_position) {
    const int pair2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int token = blockIdx.y;
    const int logical_head = blockIdx.z;

    const int half_rotary_dim = rotary_dim / 2;
    const int total_heads = num_q_heads + num_kv_heads;
    if (token >= num_tokens || logical_head >= total_heads || pair2 + 1 >= half_rotary_dim) return;

    const bool is_q = logical_head < num_q_heads;
    const int head = is_q ? logical_head : logical_head - num_q_heads;
    const int num_heads = is_q ? num_q_heads : num_kv_heads;
    __nv_bfloat16* x = is_q ? q : k;

    const int pos = positions[token];
    if (pos < 0 || pos >= rope_cache_max_position) return;

    const int64_t base = (static_cast<int64_t>(token) * num_heads + head) * head_dim;

    __nv_bfloat16* lo_ptr = x + base + pair2;
    __nv_bfloat16* hi_ptr = x + base + half_rotary_dim + pair2;
    const __nv_bfloat162 lo_h = *reinterpret_cast<const __nv_bfloat162*>(lo_ptr);
    const __nv_bfloat162 hi_h = *reinterpret_cast<const __nv_bfloat162*>(hi_ptr);
    const float2 lo = __bfloat1622float2(lo_h);
    const float2 hi = __bfloat1622float2(hi_h);

    const float2 cs0 = rope_cache[static_cast<int64_t>(pos) * half_rotary_dim + pair2];
    const float2 cs1 = rope_cache[static_cast<int64_t>(pos) * half_rotary_dim + pair2 + 1];

    *reinterpret_cast<__nv_bfloat162*>(lo_ptr) =
        make_bfloat162(__float2bfloat16_rn(lo.x * cs0.x - hi.x * cs0.y),
                       __float2bfloat16_rn(lo.y * cs1.x - hi.y * cs1.y));
    *reinterpret_cast<__nv_bfloat162*>(hi_ptr) =
        make_bfloat162(__float2bfloat16_rn(hi.x * cs0.x + lo.x * cs0.y),
                       __float2bfloat16_rn(hi.y * cs1.x + lo.y * cs1.y));
}

}  // namespace

Result<void> launch_rope(__nv_bfloat16* q, __nv_bfloat16* k, const int32_t* positions,
                         const float2* rope_cache, int num_tokens, int num_q_heads,
                         int num_kv_heads, int head_dim, int rotary_dim,
                         int rope_cache_max_position, cudaStream_t stream) {
    const int half_rotary_dim = rotary_dim / 2;
    const int total_heads = num_q_heads + num_kv_heads;
    constexpr int kThreads = 64;

    bool use_bf162 = (rotary_dim % 4 == 0) && (head_dim % 2 == 0) && is_aligned_4(q) &&
                     is_aligned_4(k) && is_aligned_8(rope_cache);

    if (use_bf162) {
        int n2 = half_rotary_dim / 2;
        dim3 grid(ceil_div(n2, kThreads), num_tokens, total_heads);
        rope_split_half_bf162_kernel<<<grid, kThreads, 0, stream>>>(
            q, k, positions, rope_cache, num_tokens, num_q_heads, num_kv_heads, head_dim,
            rotary_dim, rope_cache_max_position);
    } else {
        dim3 grid(ceil_div(half_rotary_dim, kThreads), num_tokens, total_heads);
        rope_split_half_scalar_kernel<<<grid, kThreads, 0, stream>>>(
            q, k, positions, rope_cache, num_tokens, num_q_heads, num_kv_heads, head_dim,
            rotary_dim, rope_cache_max_position);
    }
    return cuda_check_last_error();
}

}  // namespace engine
}  // namespace ccinfer
