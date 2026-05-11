#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "engine/backend/cuda/cuda_utils.h"
#include "engine/kernel/cuda_kernels.h"

namespace ccinfer {
namespace engine {
namespace {

// dst[i] = dst[i] + src[i] (residual add)
__global__ void element_add_kernel(__nv_bfloat16* dst, const __nv_bfloat16* src, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const float a = __bfloat162float(dst[i]);
    const float b = __bfloat162float(src[i]);

    dst[i] = __float2bfloat16_rn(a + b);
}

// Input:
//   qkv_out: [T, qkv_dim] where qkv_dim = (nq + 2*nkv) * hd
//   Per token layout: qkv_out[t] = [Q_flat | K_flat | V_flat]
//
// Output:
//   q: [T, nq,  hd]
//   k: [T, nkv, hd]
//   v: [T, nkv, hd]
__global__ void split_qkv_kernel(const __nv_bfloat16* __restrict__ qkv,
                                 __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
                                 __nv_bfloat16* __restrict__ v, int T, int nq, int nkv, int hd) {
    const int q_dim = nq * hd;
    const int k_dim = nkv * hd;
    const int v_dim = nkv * hd;
    const int qkv_dim = q_dim + k_dim + v_dim;

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const int64_t total = static_cast<int64_t>(T) * qkv_dim;

    if (idx >= total) {
        return;
    }

    const int token = static_cast<int>(idx / qkv_dim);
    const int off = static_cast<int>(idx % qkv_dim);

    const __nv_bfloat16 val = qkv[idx];

    if (off < q_dim) {
        q[static_cast<int64_t>(token) * q_dim + off] = val;
    } else if (off < q_dim + k_dim) {
        const int k_off = off - q_dim;
        k[static_cast<int64_t>(token) * k_dim + k_off] = val;
    } else {
        const int v_off = off - q_dim - k_dim;
        v[static_cast<int64_t>(token) * v_dim + v_off] = val;
    }
}

// Gather embedding rows: input_embeds[t, d] = embed_table[token_ids[t], d]
__global__ void embed_kernel(const __nv_bfloat16* embed_table, const int32_t* token_ids,
                             __nv_bfloat16* input_embeds, int num_tokens, int d_model) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(num_tokens) * d_model;
    if (i >= total) return;

    const int t = static_cast<int>(i / d_model);
    const int d = static_cast<int>(i % d_model);
    const int64_t row = static_cast<int64_t>(token_ids[t]);

    input_embeds[i] = embed_table[row * d_model + d];
}

}  // namespace

Result<void> launch_element_add(__nv_bfloat16* dst, const __nv_bfloat16* src, int64_t n,
                                cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    const int grid = static_cast<int>((n + kBlockSize - 1) / kBlockSize);

    element_add_kernel<<<grid, kBlockSize, 0, stream>>>(dst, src, n);

    return cuda_check_last_error();
}

Result<void> launch_split_qkv(const __nv_bfloat16* qkv, __nv_bfloat16* q, __nv_bfloat16* k,
                              __nv_bfloat16* v, int T, int nq, int nkv, int hd,
                              cudaStream_t stream) {
    const int qkv_dim = (nq + 2 * nkv) * hd;
    const int64_t total = static_cast<int64_t>(T) * qkv_dim;

    constexpr int kBlockSize = 256;
    const int grid = static_cast<int>((total + kBlockSize - 1) / kBlockSize);

    split_qkv_kernel<<<grid, kBlockSize, 0, stream>>>(qkv, q, k, v, T, nq, nkv, hd);

    return cuda_check_last_error();
}

Result<void> launch_embed(const __nv_bfloat16* embed_table, const int32_t* token_ids,
                          __nv_bfloat16* input_embeds, int num_tokens, int d_model,
                          cudaStream_t stream) {
    const int64_t total = static_cast<int64_t>(num_tokens) * d_model;

    constexpr int kBlockSize = 256;
    const int grid = static_cast<int>((total + kBlockSize - 1) / kBlockSize);

    embed_kernel<<<grid, kBlockSize, 0, stream>>>(embed_table, token_ids, input_embeds, num_tokens,
                                                  d_model);

    return cuda_check_last_error();
}

}  // namespace engine
}  // namespace ccinfer
