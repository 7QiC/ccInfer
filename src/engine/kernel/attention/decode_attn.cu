#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "engine/backend/cuda/cuda_utils.h"
#include "engine/kernel/attention/attn_common.cuh"
#include "engine/kernel/cuda_common.cuh"
#include "engine/kernel/cuda_kernels.h"

namespace ccinfer {
namespace engine {
namespace {

// Correctness-first paged decode attention with tiled scores and online softmax.
//
// One CTA computes one (request, q_head).
//
// Layout assumptions:
//   q:       [batch_size][num_q_heads][head_dim]
//   k_cache: [num_slots][num_kv_heads][head_dim]
//   v_cache: [num_slots][num_kv_heads][head_dim]
//   output:  [batch_size][num_q_heads][head_dim]
//
// Metadata:
//   block_table:  [batch_size][max_blocks_per_req]
//   context_lens: [batch_size]
//
// Notes:
//   - context_lens[req] is the number of visible KV tokens for this decode step.
//   - This is not a tensor-core FlashAttention implementation.
//   - It is a compact, correct, maintainable baseline using:
//       * paged KV addressing
//       * warp-parallel QK score computation
//       * tiled score buffering
//       * online softmax
//       * no full-score materialization

template <int kThreadsPerCTA, int kTileTokens>
__global__ void decode_attention_kernel(const __nv_bfloat16* __restrict__ q,
                                        const __nv_bfloat16* __restrict__ k_cache,
                                        const __nv_bfloat16* __restrict__ v_cache,
                                        const int32_t* __restrict__ block_table,
                                        const int32_t* __restrict__ context_lens,
                                        __nv_bfloat16* __restrict__ output, int batch_size,
                                        int max_blocks_per_req, int num_q_heads, int num_kv_heads,
                                        int head_dim, int cache_block_size, float scale) {
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerCTA = kThreadsPerCTA / kWarpSize;

    const int req_idx = blockIdx.x;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;

    if (req_idx >= batch_size || q_head >= num_q_heads) {
        return;
    }

    extern __shared__ __align__(16) unsigned char smem_raw[];

    // Dynamic shared memory layout:
    //   q_s:        [head_dim] bf16
    //   scores:     [kTileTokens] float
    //   kv_offsets: [kTileTokens] int64_t
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    const std::size_t q_bytes = static_cast<std::size_t>(head_dim) * sizeof(__nv_bfloat16);
    const std::size_t scores_offset = align_up(q_bytes, alignof(float));

    float* scores = reinterpret_cast<float*>(smem_raw + scores_offset);

    const std::size_t scores_bytes = static_cast<std::size_t>(kTileTokens) * sizeof(float);
    const std::size_t offsets_offset = align_up(scores_offset + scores_bytes, alignof(int64_t));

    int64_t* kv_offsets = reinterpret_cast<int64_t*>(smem_raw + offsets_offset);

    __nv_bfloat16* out_vec =
        output + (static_cast<int64_t>(req_idx) * num_q_heads + q_head) * head_dim;

    const int context_len = context_lens[req_idx];

    if (context_len <= 0) {
        if (tid < head_dim) {
            out_vec[tid] = __float2bfloat16_rn(0.0f);
        }
        return;
    }

    const int num_blocks = (context_len + cache_block_size - 1) / cache_block_size;

    // Avoid out-of-bound block table access on invalid metadata.
    if (num_blocks > max_blocks_per_req) {
        if (tid < head_dim) {
            out_vec[tid] = __float2bfloat16_rn(0.0f);
        }
        return;
    }

    const int gqa = num_q_heads / num_kv_heads;
    const int kv_head = q_head / gqa;

    if (kv_head < 0 || kv_head >= num_kv_heads) {
        return;
    }

    const __nv_bfloat16* q_vec =
        q + (static_cast<int64_t>(req_idx) * num_q_heads + q_head) * head_dim;

    const int32_t* req_blocks = block_table + static_cast<int64_t>(req_idx) * max_blocks_per_req;

    // Cache Q in shared memory. It is reused for every K tile.
    for (int d = tid; d < head_dim; d += kThreadsPerCTA) {
        q_s[d] = q_vec[d];
    }
    __syncthreads();

    const int out_dim = tid;
    const bool has_out_dim = out_dim < head_dim;

    // Online softmax state.
    //
    // m   = running max score
    // l   = sum_j exp(score_j - m)
    // acc = sum_j exp(score_j - m) * V_j[out_dim]
    float m = -FLT_MAX;
    float l = 0.0f;
    float acc = 0.0f;

    for (int tile_start = 0; tile_start < context_len; tile_start += kTileTokens) {
        const int remaining = context_len - tile_start;
        const int tile_len = remaining < kTileTokens ? remaining : kTileTokens;

        // Compute QK scores for this tile.
        //
        // Each warp computes one token score at a time.
        // With 256 threads, 8 warps compute 8 token scores in parallel.
        for (int tile_base = 0; tile_base < tile_len; tile_base += kWarpsPerCTA) {
            const int tile_i = tile_base + warp_id;
            const int ctx_pos = tile_start + tile_i;

            bool valid = false;
            int64_t kv_base = -1;

            if (tile_i < tile_len) {
                valid = get_kv_cache_base_offset(ctx_pos, kv_head, head_dim, num_kv_heads,
                                                 cache_block_size, max_blocks_per_req, req_blocks,
                                                 &kv_base);
            }

            float partial = 0.0f;

            if (valid) {
                const __nv_bfloat16* k_vec = k_cache + kv_base;

                for (int d = lane; d < head_dim; d += kWarpSize) {
                    const float q_val = __bfloat162float(q_s[d]);
                    const float k_val = __bfloat162float(k_vec[d]);
                    partial += q_val * k_val;
                }
            }

            const float dot = warp_reduce_sum(partial);

            if (lane == 0 && tile_i < tile_len) {
                scores[tile_i] = valid ? dot * scale : -FLT_MAX;
                kv_offsets[tile_i] = valid ? kv_base : -1;
            }
        }

        __syncthreads();

        // Compute tile max.
        float local_tile_max = -FLT_MAX;
        for (int i = tid; i < tile_len; i += kThreadsPerCTA) {
            local_tile_max = fmaxf(local_tile_max, scores[i]);
        }

        const float tile_max = block_reduce_max<kThreadsPerCTA>(local_tile_max);

        // Entire tile invalid. Skip it.
        if (tile_max == -FLT_MAX) {
            __syncthreads();
            continue;
        }

        const float new_m = fmaxf(m, tile_max);
        const float alpha = (m == -FLT_MAX) ? 0.0f : expf(m - new_m);

        // Compute tile denominator under new_m.
        float local_tile_l = 0.0f;
        for (int i = tid; i < tile_len; i += kThreadsPerCTA) {
            local_tile_l += expf(scores[i] - new_m);
        }

        const float tile_l = block_reduce_sum<kThreadsPerCTA>(local_tile_l);

        // Accumulate weighted V for this thread's output dimension.
        float tile_acc = 0.0f;

        if (has_out_dim) {
            for (int i = 0; i < tile_len; ++i) {
                const int64_t kv_base = kv_offsets[i];

                if (kv_base >= 0) {
                    const float p = expf(scores[i] - new_m);
                    const float v_val = __bfloat162float(v_cache[kv_base + out_dim]);
                    tile_acc += p * v_val;
                }
            }

            acc = acc * alpha + tile_acc;
        }

        l = l * alpha + tile_l;
        m = new_m;

        __syncthreads();
    }

    if (has_out_dim) {
        const float out_val = l > 0.0f ? acc / l : 0.0f;
        out_vec[out_dim] = __float2bfloat16_rn(out_val);
    }
}

}  // namespace

Result<void> launch_decode_attention(const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
                                     const __nv_bfloat16* v_cache, const int32_t* block_table,
                                     const int32_t* context_lens, __nv_bfloat16* output,
                                     int batch_size, int max_blocks_per_req, int num_q_heads,
                                     int num_kv_heads, int head_dim, int cache_block_size,
                                     cudaStream_t stream) {
    constexpr int kThreadsPerCTA = 256;
    constexpr int kTileTokens = 64;

    static_assert(kThreadsPerCTA % 32 == 0);
    static_assert(kTileTokens > 0);

    // This kernel maps one output dimension to one CUDA thread.
    // For usual LLM head_dim values such as 64, 80, 96, 128, 256 this is fine.
    // For head_dim > 256, use a multi-output-per-thread or multi-CTA variant.
    if (head_dim > kThreadsPerCTA) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    const std::size_t q_bytes = static_cast<std::size_t>(head_dim) * sizeof(__nv_bfloat16);
    const std::size_t scores_offset = align_up(q_bytes, alignof(float));

    const std::size_t scores_bytes = static_cast<std::size_t>(kTileTokens) * sizeof(float);
    const std::size_t offsets_offset = align_up(scores_offset + scores_bytes, alignof(int64_t));

    const std::size_t shared_bytes =
        offsets_offset + static_cast<std::size_t>(kTileTokens) * sizeof(int64_t);

    dim3 grid(batch_size, num_q_heads);
    dim3 block(kThreadsPerCTA);

    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    decode_attention_kernel<kThreadsPerCTA, kTileTokens><<<grid, block, shared_bytes, stream>>>(
        q, k_cache, v_cache, block_table, context_lens, output, batch_size, max_blocks_per_req,
        num_q_heads, num_kv_heads, head_dim, cache_block_size, scale);

    return cuda_check_last_error();
}

}  // namespace engine
}  // namespace ccinfer
