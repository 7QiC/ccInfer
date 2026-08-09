#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "backend/cuda/cuda_utils.h"
#include "kernel/cuda_kernels.h"

namespace ccinfer {
namespace {

// Decode attention over the paged KV cache, with online softmax.
// Layout matches the current write_kv_cache contract (no transposed K/V
// storage):
//
//   q:       [batch_size, num_q_heads, head_dim]
//   k_cache: [num_slots,  num_kv_heads, head_dim]
//   v_cache: [num_slots,  num_kv_heads, head_dim]
//   output:  [batch_size, num_q_heads, head_dim]

template <int kBlockSize>
__device__ __forceinline__ bool get_base_offset(int ctx_pos, int kv_head, int head_dim,
                                                int num_kv_heads, int max_blocks_per_req,
                                                const int32_t* __restrict__ req_blocks,
                                                int64_t* out_base_offset) {
    const int logical_block = ctx_pos / kBlockSize;
    const int offset_in_block = ctx_pos - logical_block * kBlockSize;
    if (logical_block < 0 || logical_block >= max_blocks_per_req) return false;

    const int physical_block = req_blocks[logical_block];
    if (physical_block < 0) return false;

    const int64_t slot = static_cast<int64_t>(physical_block) * kBlockSize + offset_in_block;
    *out_base_offset = (slot * num_kv_heads + kv_head) * static_cast<int64_t>(head_dim);
    return true;
}

template <int kNumWarps>
__device__ __forceinline__ float reduce_block_sum(float* smem, float value) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

#pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, mask);
    }
    if (lane == 0) smem[warp] = value;
    __syncthreads();

    value = lane < kNumWarps ? smem[lane] : 0.0f;
#pragma unroll
    for (int mask = kNumWarps / 2; mask >= 1; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, mask);
    }
    value = __shfl_sync(0xffffffffu, value, 0);
    __syncthreads();
    return value;
}

template <int kNumWarps>
__device__ __forceinline__ float reduce_block_max(float* smem, float value) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

#pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
    }
    if (lane == 0) smem[warp] = value;
    __syncthreads();

    value = lane < kNumWarps ? smem[lane] : -FLT_MAX;
#pragma unroll
    for (int mask = kNumWarps / 2; mask >= 1; mask >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
    }
    value = __shfl_sync(0xffffffffu, value, 0);
    __syncthreads();
    return value;
}

template <int kHeadDim, int kBlockSize, int kNumThreads>
__global__ void decode_attention_kernel(const __nv_bfloat16* __restrict__ q,
                                        const __nv_bfloat16* __restrict__ k_cache,
                                        const __nv_bfloat16* __restrict__ v_cache,
                                        const int32_t* __restrict__ block_table,
                                        const int32_t* __restrict__ context_lens,
                                        __nv_bfloat16* __restrict__ output, int batch_size,
                                        int max_blocks_per_req, int num_q_heads, int num_kv_heads,
                                        float scale) {
    constexpr int kWarpSize = 32;
    constexpr int kNumWarps = kNumThreads / kWarpSize;
    static_assert(kNumThreads % kWarpSize == 0);
    static_assert(kBlockSize == 16);

    const int req_idx = blockIdx.x;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    if (req_idx >= batch_size || q_head >= num_q_heads) return;

    __nv_bfloat16* out_vec =
        output + (static_cast<int64_t>(req_idx) * num_q_heads + q_head) * kHeadDim;

    const int context_len = context_lens[req_idx];
    if (context_len <= 0) {
        for (int d = tid; d < kHeadDim; d += kNumThreads) {
            out_vec[d] = __float2bfloat16_rn(0.0f);
        }
        return;
    }

    const int num_blocks = (context_len + kBlockSize - 1) / kBlockSize;
    if (num_blocks > max_blocks_per_req) {
        for (int d = tid; d < kHeadDim; d += kNumThreads) {
            out_vec[d] = __float2bfloat16_rn(0.0f);
        }
        return;
    }

    const int gqa = num_q_heads / num_kv_heads;
    const int kv_head = q_head / gqa;
    if (kv_head < 0 || kv_head >= num_kv_heads) return;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* logits = reinterpret_cast<float*>(smem_raw);
    __shared__ float reduce_smem[kNumWarps];

    const __nv_bfloat16* q_vec =
        q + (static_cast<int64_t>(req_idx) * num_q_heads + q_head) * kHeadDim;
    const int32_t* req_blocks = block_table + static_cast<int64_t>(req_idx) * max_blocks_per_req;

    float qk_max = -FLT_MAX;

    for (int token_idx = warp_id; token_idx < context_len; token_idx += kNumWarps) {
        float qk = -FLT_MAX;

        int64_t k_base = 0;
        const bool valid =
            get_base_offset<kBlockSize>(token_idx, kv_head, kHeadDim, num_kv_heads,
                                        max_blocks_per_req, req_blocks, &k_base);

        if (valid) {
            float partial = 0.0f;
            for (int d = lane; d < kHeadDim; d += kWarpSize) {
                partial += __bfloat162float(q_vec[d]) * __bfloat162float(k_cache[k_base + d]);
            }
#pragma unroll
            for (int mask = 16; mask >= 1; mask >>= 1) {
                partial += __shfl_xor_sync(0xffffffffu, partial, mask);
            }
            qk = partial * scale;
        }

        if (lane == 0) {
            logits[token_idx] = qk;
            qk_max = fmaxf(qk_max, qk);
        }
    }

    qk_max = reduce_block_max<kNumWarps>(reduce_smem, qk_max);

    float exp_sum = 0.0f;
    for (int i = tid; i < context_len; i += kNumThreads) {
        const float p = __expf(logits[i] - qk_max);
        logits[i] = p;
        exp_sum += p;
    }
    exp_sum = reduce_block_sum<kNumWarps>(reduce_smem, exp_sum);

    const float inv_sum = __fdividef(1.0f, exp_sum + 1e-6f);
    for (int i = tid; i < context_len; i += kNumThreads) {
        logits[i] *= inv_sum;
    }
    __syncthreads();

    for (int d = tid; d < kHeadDim; d += kNumThreads) {
        float acc = 0.0f;
        for (int token_idx = 0; token_idx < context_len; ++token_idx) {
            int64_t v_base = 0;
            const bool valid = get_base_offset<kBlockSize>(
                token_idx, kv_head, kHeadDim, num_kv_heads, max_blocks_per_req, req_blocks,
                &v_base);
            if (valid) {
                acc += logits[token_idx] * __bfloat162float(v_cache[v_base + d]);
            }
        }
        out_vec[d] = __float2bfloat16_rn(acc);
    }
}

template <int kHeadDim>
Result<void> launch_decode_attention_typed(const __nv_bfloat16* q,
                                           const __nv_bfloat16* k_cache,
                                           const __nv_bfloat16* v_cache,
                                           const int32_t* block_table,
                                           const int32_t* context_lens,
                                           __nv_bfloat16* output, int batch_size,
                                           int max_blocks_per_req, int num_q_heads,
                                           int num_kv_heads, int cache_block_size,
                                           cudaStream_t stream) {
    constexpr int kBlockSize = 16;
    constexpr int kNumThreads = 128;
    if (cache_block_size != kBlockSize) return std::unexpected(ErrorCode::InvalidArgument);

    const std::size_t shared_bytes =
        static_cast<std::size_t>(max_blocks_per_req) * kBlockSize * sizeof(float);
    if (shared_bytes > 96 * 1024) return std::unexpected(ErrorCode::InvalidArgument);

    auto kernel = decode_attention_kernel<kHeadDim, kBlockSize, kNumThreads>;
    if (shared_bytes > 48 * 1024) {
        auto r = cuda_check(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
        if (!r) return r;
    }

    dim3 grid(batch_size, num_q_heads);
    dim3 block(kNumThreads);
    const float scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));

    decode_attention_kernel<kHeadDim, kBlockSize, kNumThreads><<<grid, block, shared_bytes, stream>>>(
        q, k_cache, v_cache, block_table, context_lens, output, batch_size, max_blocks_per_req,
        num_q_heads, num_kv_heads, scale);

    return cuda_check_last_error();
}

}  // namespace

Result<void> launch_decode_attention(const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
                                     const __nv_bfloat16* v_cache, const int32_t* block_table,
                                     const int32_t* context_lens, __nv_bfloat16* output,
                                     int batch_size, int max_blocks_per_req, int num_q_heads,
                                     int num_kv_heads, int head_dim, int cache_block_size,
                                     cudaStream_t stream) {
    switch (head_dim) {
        case 32:
            return launch_decode_attention_typed<32>(q, k_cache, v_cache, block_table,
                                                     context_lens, output, batch_size,
                                                     max_blocks_per_req, num_q_heads,
                                                     num_kv_heads, cache_block_size, stream);
        case 64:
            return launch_decode_attention_typed<64>(q, k_cache, v_cache, block_table,
                                                     context_lens, output, batch_size,
                                                     max_blocks_per_req, num_q_heads,
                                                     num_kv_heads, cache_block_size, stream);
        case 80:
            return launch_decode_attention_typed<80>(q, k_cache, v_cache, block_table,
                                                     context_lens, output, batch_size,
                                                     max_blocks_per_req, num_q_heads,
                                                     num_kv_heads, cache_block_size, stream);
        case 96:
            return launch_decode_attention_typed<96>(q, k_cache, v_cache, block_table,
                                                     context_lens, output, batch_size,
                                                     max_blocks_per_req, num_q_heads,
                                                     num_kv_heads, cache_block_size, stream);
        case 112:
            return launch_decode_attention_typed<112>(q, k_cache, v_cache, block_table,
                                                      context_lens, output, batch_size,
                                                      max_blocks_per_req, num_q_heads,
                                                      num_kv_heads, cache_block_size, stream);
        case 120:
            return launch_decode_attention_typed<120>(q, k_cache, v_cache, block_table,
                                                      context_lens, output, batch_size,
                                                      max_blocks_per_req, num_q_heads,
                                                      num_kv_heads, cache_block_size, stream);
        case 128:
            return launch_decode_attention_typed<128>(q, k_cache, v_cache, block_table,
                                                      context_lens, output, batch_size,
                                                      max_blocks_per_req, num_q_heads,
                                                      num_kv_heads, cache_block_size, stream);
        case 192:
            return launch_decode_attention_typed<192>(q, k_cache, v_cache, block_table,
                                                      context_lens, output, batch_size,
                                                      max_blocks_per_req, num_q_heads,
                                                      num_kv_heads, cache_block_size, stream);
        case 256:
            return launch_decode_attention_typed<256>(q, k_cache, v_cache, block_table,
                                                      context_lens, output, batch_size,
                                                      max_blocks_per_req, num_q_heads,
                                                      num_kv_heads, cache_block_size, stream);
        default:
            return std::unexpected(ErrorCode::InvalidArgument);
    }
}

}  // namespace ccinfer
