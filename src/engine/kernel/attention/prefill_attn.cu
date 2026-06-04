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

// Correctness-first tiled paged prefill attention with online softmax.
//
// Layout assumptions:
//   q:       [total_query_tokens][num_q_heads][head_dim]
//   k_cache: [num_slots][num_kv_heads][head_dim]
//   v_cache: [num_slots][num_kv_heads][head_dim]
//   output:  [total_query_tokens][num_q_heads][head_dim]
//
// Metadata:
//   block_table:     [batch_size][max_blocks_per_req]
//   query_start_loc: [batch_size + 1], device pointer
//   context_lens:    [batch_size], device pointer
//
// Semantics:
//   context_lens[req] = prefix_len + query_len.
//   For query token i in this request, visible context length is:
//      prefix_len + i + 1
//
// This is not FlashAttention-2. It is a maintainable baseline that uses:
//   - K tiling
//   - online softmax
//   - no full-score materialization
//   - paged KV cache addressing
//   - causal prefill masking

template <int kNumThreads, int kTileTokens>
__global__ void prefill_attention_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ query_start_loc, const int32_t* __restrict__ context_lens,
    __nv_bfloat16* __restrict__ output, int batch_size, int total_query_tokens,
    int max_blocks_per_req, int num_q_heads, int num_kv_heads, int head_dim, int cache_block_size,
    float scale) {
    const int global_query_idx = blockIdx.x;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x;

    if (global_query_idx >= total_query_tokens || q_head >= num_q_heads) {
        return;
    }

    // Dynamic shared memory layout:
    //   q_s:       [head_dim]
    //   k_s:       [kTileTokens][head_dim]
    //   scores:    [kTileTokens]
    //   kv_offsets:[kTileTokens]
    extern __shared__ __align__(16) unsigned char smem_raw[];

    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* k_s = q_s + head_dim;

    const std::size_t bf16_bytes =
        static_cast<std::size_t>(1 + kTileTokens) * head_dim * sizeof(__nv_bfloat16);
    const std::size_t score_offset = align_up(bf16_bytes, alignof(float));

    float* scores = reinterpret_cast<float*>(smem_raw + score_offset);
    const std::size_t scores_bytes = static_cast<std::size_t>(kTileTokens) * sizeof(float);
    const std::size_t offsets_offset = align_up(score_offset + scores_bytes, alignof(int64_t));
    int64_t* kv_offsets = reinterpret_cast<int64_t*>(smem_raw + offsets_offset);

    // Find request id for this global query token.
    // This is O(batch_size), acceptable for a baseline.
    // Later optimization: precompute query_to_request[total_query_tokens].
    int req_idx = 0;
    while (req_idx + 1 < batch_size && query_start_loc[req_idx + 1] <= global_query_idx) {
        ++req_idx;
    }

    if (req_idx < 0 || req_idx >= batch_size) {
        return;
    }

    const int query_begin = query_start_loc[req_idx];
    const int query_end = query_start_loc[req_idx + 1];
    const int query_len = query_end - query_begin;
    const int local_query_idx = global_query_idx - query_begin;

    if (query_len <= 0 || local_query_idx < 0 || local_query_idx >= query_len) {
        return;
    }

    const int context_len = context_lens[req_idx];

    // context_len should include both prefix and current prefill tokens.
    const int prefix_len = context_len - query_len;
    if (prefix_len < 0) {
        return;
    }

    // Causal prefill mask.
    const int effective_context_len = prefix_len + local_query_idx + 1;
    if (effective_context_len <= 0 || effective_context_len > context_len) {
        return;
    }

    const int gqa = num_q_heads / num_kv_heads;
    const int kv_head = q_head / gqa;
    if (kv_head < 0 || kv_head >= num_kv_heads) {
        return;
    }

    const int32_t* req_blocks = block_table + static_cast<int64_t>(req_idx) * max_blocks_per_req;

    const __nv_bfloat16* q_vec =
        q + (static_cast<int64_t>(global_query_idx) * num_q_heads + q_head) * head_dim;

    __nv_bfloat16* out_vec =
        output + (static_cast<int64_t>(global_query_idx) * num_q_heads + q_head) * head_dim;

    // Cache Q in shared memory. Reused by all K tiles.
    for (int d = tid; d < head_dim; d += kNumThreads) {
        q_s[d] = q_vec[d];
    }
    __syncthreads();

    // This baseline assigns one output dimension to one thread.
    // Host enforces head_dim <= kNumThreads.
    const int out_dim = tid;
    const bool has_output_dim = out_dim < head_dim;

    // Online softmax state.
    // acc stores unnormalized sum: sum_j exp(score_j - m) * V_j[d]
    // l stores denominator:       sum_j exp(score_j - m)
    // m stores running max.
    float m = -FLT_MAX;
    float l = 0.0f;
    float acc = 0.0f;

    for (int tile_start = 0; tile_start < effective_context_len; tile_start += kTileTokens) {
        const int remaining = effective_context_len - tile_start;
        const int tile_len = remaining < kTileTokens ? remaining : kTileTokens;

        // Load K tile to shared memory.
        const int tile_elems = tile_len * head_dim;
        for (int linear = tid; linear < tile_elems; linear += kNumThreads) {
            const int tile_i = linear / head_dim;
            const int dim = linear - tile_i * head_dim;
            const int ctx_pos = tile_start + tile_i;

            int64_t k_base = 0;
            const bool valid =
                get_kv_cache_base_offset(ctx_pos, kv_head, head_dim, num_kv_heads, cache_block_size,
                                         max_blocks_per_req, req_blocks, &k_base);
            if (dim == 0) {
                kv_offsets[tile_i] = valid ? k_base : -1;
            }

            k_s[linear] = valid ? k_cache[k_base + dim] : __float2bfloat16_rn(0.0f);
        }
        __syncthreads();

        // Compute scaled QK scores for this tile.
        for (int tile_i = 0; tile_i < tile_len; ++tile_i) {
            const bool valid = kv_offsets[tile_i] >= 0;

            float partial = 0.0f;

            if (valid) {
                for (int d = tid; d < head_dim; d += kNumThreads) {
                    const float q_val = __bfloat162float(q_s[d]);
                    const float k_val = __bfloat162float(k_s[tile_i * head_dim + d]);
                    partial += q_val * k_val;
                }
            }

            const float dot = block_reduce_sum<kNumThreads>(partial);

            if (tid == 0) {
                scores[tile_i] = valid ? dot * scale : -FLT_MAX;
            }
            __syncthreads();
        }

        // Compute tile max.
        float local_tile_max = -FLT_MAX;
        for (int i = tid; i < tile_len; i += kNumThreads) {
            local_tile_max = fmaxf(local_tile_max, scores[i]);
        }

        const float tile_max = block_reduce_max<kNumThreads>(local_tile_max);

        // If every position in this tile is invalid, skip the online update.
        if (tile_max == -FLT_MAX) {
            __syncthreads();
            continue;
        }

        const float new_m = fmaxf(m, tile_max);
        const float alpha = (m == -FLT_MAX) ? 0.0f : expf(m - new_m);

        // Compute tile softmax denominator under new_m.
        float local_tile_l = 0.0f;
        for (int i = tid; i < tile_len; i += kNumThreads) {
            local_tile_l += expf(scores[i] - new_m);
        }

        const float tile_l = block_reduce_sum<kNumThreads>(local_tile_l);

        // Compute tile contribution to the output dimension assigned to this thread.
        float tile_acc = 0.0f;

        if (has_output_dim) {
            for (int tile_i = 0; tile_i < tile_len; ++tile_i) {
                const int64_t v_base = kv_offsets[tile_i];

                if (v_base >= 0) {
                    const float p = expf(scores[tile_i] - new_m);
                    const float v_val = __bfloat162float(v_cache[v_base + out_dim]);
                    tile_acc += p * v_val;
                }
            }

            acc = acc * alpha + tile_acc;
        }

        l = l * alpha + tile_l;
        m = new_m;

        __syncthreads();
    }

    if (has_output_dim) {
        const float out = l > 0.0f ? acc / l : 0.0f;
        out_vec[out_dim] = __float2bfloat16_rn(out);
    }
}

}  // namespace

Result<void> launch_prefill_attention(const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
                                      const __nv_bfloat16* v_cache, const int32_t* block_table,
                                      const int32_t* query_start_loc, const int32_t* context_lens,
                                      __nv_bfloat16* output, int batch_size, int total_query_tokens,
                                      int max_blocks_per_req, int num_q_heads, int num_kv_heads,
                                      int head_dim, int cache_block_size, cudaStream_t stream) {
    constexpr int kNumThreads = 256;
    constexpr int kTileTokens = 64;

    // This implementation maps one output dim to one CUDA thread.
    // Keep this explicit. For head_dim > 256, use a different kernel.
    if (head_dim > kNumThreads) {
        return std::unexpected(ErrorCode::InvalidArgument);
    }

    dim3 grid(total_query_tokens, num_q_heads);
    dim3 block(kNumThreads);

    const std::size_t bf16_bytes =
        static_cast<std::size_t>(1 + kTileTokens) * head_dim * sizeof(__nv_bfloat16);
    const std::size_t score_offset = align_up(bf16_bytes, alignof(float));
    const std::size_t scores_bytes = kTileTokens * sizeof(float);
    const std::size_t offsets_offset = align_up(score_offset + scores_bytes, alignof(int64_t));
    const std::size_t shared_bytes = offsets_offset + kTileTokens * sizeof(int64_t);

    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    prefill_attention_kernel<kNumThreads, kTileTokens><<<grid, block, shared_bytes, stream>>>(
        q, k_cache, v_cache, block_table, query_start_loc, context_lens, output, batch_size,
        total_query_tokens, max_blocks_per_req, num_q_heads, num_kv_heads, head_dim,
        cache_block_size, scale);

    return cuda_check_last_error();
}

}  // namespace engine
}  // namespace ccinfer
