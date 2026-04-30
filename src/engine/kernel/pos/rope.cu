#include "engine/kernel/pos/rope.h"

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "backend/cuda/cuda_utils.h"

namespace ccinfer {
namespace engine {

namespace {

inline bool is_aligned_4(const void* ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 0x3u) == 0; }

inline bool is_aligned_8(const void* ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 0x7u) == 0; }

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// -----------------------------------------------------------------------------
// Scalar fallback.
//
// Layout:
//   x: [num_tokens, num_heads, head_dim]
//
// Split-half RoPE:
//   rotary_dim = head_dim for Llama-like Phase 2
//   half_rotary_dim = rotary_dim / 2
//
// For each pair index i:
//   x0 = x[i]
//   x1 = x[i + half_rotary_dim]
//
//   y0 = x0 * cos - x1 * sin
//   y1 = x1 * cos + x0 * sin
// -----------------------------------------------------------------------------
__global__ void rope_split_half_scalar_kernel(half* __restrict__ q, half* __restrict__ k,
                                              const int32_t* __restrict__ positions,
                                              const float2* __restrict__ rope_cache, int num_tokens,
                                              int num_q_heads, int num_kv_heads, int head_dim,
                                              int rotary_dim, int max_position) {
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    const int logical_head = blockIdx.z;

    const int half_rotary_dim = rotary_dim / 2;
    const int total_heads = num_q_heads + num_kv_heads;

    if (token >= num_tokens || logical_head >= total_heads || pair_idx >= half_rotary_dim) {
        return;
    }

    const bool is_q = logical_head < num_q_heads;
    const int head = is_q ? logical_head : logical_head - num_q_heads;
    const int num_heads = is_q ? num_q_heads : num_kv_heads;
    half* x = is_q ? q : k;

    const int pos = positions[token];
    if (pos < 0 || pos >= max_position) {
        return;
    }

    const int64_t base = (static_cast<int64_t>(token) * num_heads + head) * head_dim;

    const int i0 = pair_idx;
    const int i1 = pair_idx + half_rotary_dim;

    const float2 cs = rope_cache[static_cast<int64_t>(pos) * half_rotary_dim + pair_idx];

    const float x0 = __half2float(x[base + i0]);
    const float x1 = __half2float(x[base + i1]);

    const float y0 = x0 * cs.x - x1 * cs.y;
    const float y1 = x1 * cs.x + x0 * cs.y;

    x[base + i0] = __float2half_rn(y0);
    x[base + i1] = __float2half_rn(y1);
}

// -----------------------------------------------------------------------------
// half2 fast path.
//
// Each thread handles two RoPE pairs:
//
//   pair0 = pair2
//   pair1 = pair2 + 1
//
// For split-half layout, we load:
//
//   lo = [x[pair0], x[pair1]]
//   hi = [x[pair0 + half_rotary_dim], x[pair1 + half_rotary_dim]]
//
// This gives vectorized half2 load/store while preserving split-half semantics.
// -----------------------------------------------------------------------------
__global__ void rope_split_half_half2_kernel(half* __restrict__ q, half* __restrict__ k,
                                             const int32_t* __restrict__ positions,
                                             const float2* __restrict__ rope_cache, int num_tokens,
                                             int num_q_heads, int num_kv_heads, int head_dim,
                                             int rotary_dim, int max_position) {
    const int pair2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int token = blockIdx.y;
    const int logical_head = blockIdx.z;

    const int half_rotary_dim = rotary_dim / 2;
    const int total_heads = num_q_heads + num_kv_heads;

    if (token >= num_tokens || logical_head >= total_heads || pair2 + 1 >= half_rotary_dim) {
        return;
    }

    const bool is_q = logical_head < num_q_heads;
    const int head = is_q ? logical_head : logical_head - num_q_heads;
    const int num_heads = is_q ? num_q_heads : num_kv_heads;
    half* x = is_q ? q : k;

    const int pos = positions[token];
    if (pos < 0 || pos >= max_position) {
        return;
    }

    const int64_t base = (static_cast<int64_t>(token) * num_heads + head) * head_dim;

    half* lo_ptr = x + base + pair2;
    half* hi_ptr = x + base + half_rotary_dim + pair2;

    const half2 lo_h = *reinterpret_cast<const half2*>(lo_ptr);
    const half2 hi_h = *reinterpret_cast<const half2*>(hi_ptr);

    const float2 lo = __half22float2(lo_h);
    const float2 hi = __half22float2(hi_h);

    const float2 cs0 = rope_cache[static_cast<int64_t>(pos) * half_rotary_dim + pair2];

    const float2 cs1 = rope_cache[static_cast<int64_t>(pos) * half_rotary_dim + pair2 + 1];

    const float y0_lo = lo.x * cs0.x - hi.x * cs0.y;
    const float y0_hi = hi.x * cs0.x + lo.x * cs0.y;

    const float y1_lo = lo.y * cs1.x - hi.y * cs1.y;
    const float y1_hi = hi.y * cs1.x + lo.y * cs1.y;

    const half2 out_lo = __floats2half2_rn(y0_lo, y1_lo);
    const half2 out_hi = __floats2half2_rn(y0_hi, y1_hi);

    *reinterpret_cast<half2*>(lo_ptr) = out_lo;
    *reinterpret_cast<half2*>(hi_ptr) = out_hi;
}

}  // namespace

void launch_rope(half* q, half* k, const int32_t* positions, const float2* rope_cache,
                 int num_tokens, int num_q_heads, int num_kv_heads, int head_dim, int rotary_dim,
                 int max_position, cudaStream_t stream) {
    if (num_tokens <= 0) {
        return;
    }

    if (q == nullptr || k == nullptr || positions == nullptr || rope_cache == nullptr) {
        return;
    }

    if (num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        return;
    }

    if (rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim % 2 != 0)) {
        return;
    }

    if (max_position <= 0) {
        return;
    }

    const int half_rotary_dim = rotary_dim / 2;
    const int total_heads = num_q_heads + num_kv_heads;

    // For common Llama head_dim=128, rotary_dim=128:
    // half_rotary_dim = 64.
    //
    // 64 threads = 2 warps. This is enough for one token-head.
    // For larger rotary_dim, grid.x creates multiple chunks.
    constexpr int kThreads = 64;

    const bool can_use_half2 = (rotary_dim % 4 == 0) && (head_dim % 2 == 0) && is_aligned_4(q) &&
                               is_aligned_4(k) && is_aligned_8(rope_cache);

    if (can_use_half2) {
        const int num_half2_pairs = half_rotary_dim / 2;
        dim3 grid(ceil_div(num_half2_pairs, kThreads), num_tokens, total_heads);
        dim3 block(kThreads);

        rope_split_half_half2_kernel<<<grid, block, 0, stream>>>(
            q, k, positions, rope_cache, num_tokens, num_q_heads, num_kv_heads, head_dim,
            rotary_dim, max_position);
    } else {
        dim3 grid(ceil_div(half_rotary_dim, kThreads), num_tokens, total_heads);
        dim3 block(kThreads);

        rope_split_half_scalar_kernel<<<grid, block, 0, stream>>>(
            q, k, positions, rope_cache, num_tokens, num_q_heads, num_kv_heads, head_dim,
            rotary_dim, max_position);
    }

    CCINFER_CUDA_CHECK(cudaGetLastError());
}

}  // namespace engine
}  // namespace ccinfer
