#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "backend/cuda/cuda_utils.h"
#include "kernel/cuda_common.cuh"
#include "kernel/cuda_kernels.h"

namespace ccinfer {

namespace {

// Token-major naive attention kernel.
//
// Public layout:
//   q:      [T, Hq,  D]
//   k:      [T, Hkv, D]
//   v:      [T, Hkv, D]
//   output: [T, Hq,  D]
//
// One CUDA block computes one (query_token, q_head).
//
// Grid:
//   blockIdx.x = query_token
//   blockIdx.y = q_head
//
// Shared memory:
//   scores[key] for key = 0..query_token
//
// Algorithm:
//   1. scores[key] = dot(q[query, q_head], k[key, kv_head]) / sqrt(D)
//   2. causal softmax over key <= query
//   3. output[query, q_head, d] = sum_key probs[key] * v[key, kv_head, d]
// -----------------------------------------------------------------------------

template <int kBlockSize>
__global__ void naive_attention_token_major_kernel(const __nv_bfloat16* __restrict__ q,
                                                   const __nv_bfloat16* __restrict__ k,
                                                   const __nv_bfloat16* __restrict__ v,
                                                   __nv_bfloat16* __restrict__ output,
                                                   int num_tokens, int n_q_heads, int n_kv_heads,
                                                   int head_dim, int gqa, float scale) {
    extern __shared__ float scores[];

    const int query_token = blockIdx.x;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x;

    if (query_token >= num_tokens || q_head >= n_q_heads) {
        return;
    }

    const int kv_head = q_head / gqa;

    if (kv_head >= n_kv_heads) {
        return;
    }

    const __nv_bfloat16* q_vec =
        q + (static_cast<int64_t>(query_token) * n_q_heads + q_head) * head_dim;

    // -------------------------------------------------------------------------
    // Step 1: compute attention scores for all valid causal keys.
    // -------------------------------------------------------------------------

    float local_max = -FLT_MAX;

    for (int key = tid; key <= query_token; key += kBlockSize) {
        const __nv_bfloat16* k_vec =
            k + (static_cast<int64_t>(key) * n_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;

        for (int d = 0; d < head_dim; ++d) {
            const float qv = __bfloat162float(q_vec[d]);
            const float kv = __bfloat162float(k_vec[d]);
            dot += qv * kv;
        }

        dot *= scale;

        scores[key] = dot;
        local_max = fmaxf(local_max, dot);
    }

    const float max_val = block_reduce_max<kBlockSize>(local_max);

    // block_reduce_max already contains synchronization, so scores are visible.

    // -------------------------------------------------------------------------
    // Step 2: softmax over key <= query_token.
    // -------------------------------------------------------------------------

    float local_sum = 0.0f;

    for (int key = tid; key <= query_token; key += kBlockSize) {
        const float e = expf(scores[key] - max_val);
        scores[key] = e;
        local_sum += e;
    }

    const float sum = block_reduce_sum<kBlockSize>(local_sum);
    const float inv_sum = 1.0f / fmaxf(sum, 1.0e-12f);

    for (int key = tid; key <= query_token; key += kBlockSize) {
        scores[key] *= inv_sum;
    }

    __syncthreads();

    // -------------------------------------------------------------------------
    // Step 3: weighted sum over V.
    //
    // Each thread computes one or more output dimensions.
    // -------------------------------------------------------------------------

    __nv_bfloat16* out_vec =
        output + (static_cast<int64_t>(query_token) * n_q_heads + q_head) * head_dim;

    for (int d = tid; d < head_dim; d += kBlockSize) {
        float acc = 0.0f;

        for (int key = 0; key <= query_token; ++key) {
            const __nv_bfloat16* v_vec =
                v + (static_cast<int64_t>(key) * n_kv_heads + kv_head) * head_dim;

            const float p = scores[key];
            const float vv = __bfloat162float(v_vec[d]);

            acc += p * vv;
        }

        out_vec[d] = __float2bfloat16_rn(acc);
    }
}

}  // namespace

Result<void> launch_naive_attention(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                    const __nv_bfloat16* v, __nv_bfloat16* output, int num_tokens,
                                    int n_q_heads, int n_kv_heads, int head_dim,
                                    cudaStream_t stream) {
    // Shared memory holds one FP32 score per token.
    // This kernel is intended for small/medium sequence correctness tests.
    const size_t shared_bytes = static_cast<size_t>(num_tokens) * sizeof(float);

    int device = 0;
    auto r = cuda_check(cudaGetDevice(&device));
    if (!r) {
        return std::unexpected(r.error());
    }

    int max_shared = 0;
    r = cuda_check(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device));
    if (!r) {
        return std::unexpected(r.error());
    }

    if (shared_bytes > static_cast<size_t>(max_shared)) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    constexpr int kBlockSize = 256;

    const int gqa = n_q_heads / n_kv_heads;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    dim3 grid(num_tokens, n_q_heads);
    dim3 block(kBlockSize);

    naive_attention_token_major_kernel<kBlockSize><<<grid, block, shared_bytes, stream>>>(
        q, k, v, output, num_tokens, n_q_heads, n_kv_heads, head_dim, gqa, scale);

    return cuda_check_last_error();
}

}  // namespace ccinfer
