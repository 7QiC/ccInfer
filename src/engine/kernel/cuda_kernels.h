#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/result.h"

namespace ccinfer {
namespace engine {

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------
Result<void> launch_rms_norm(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                             __nv_bfloat16* output, int rows, int dim, float eps,
                             cudaStream_t stream);

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------
Result<void> launch_rope(__nv_bfloat16* q, __nv_bfloat16* k, const int32_t* positions,
                         const float2* rope_cache, int num_tokens, int num_q_heads,
                         int num_kv_heads, int head_dim, int rotary_dim, int max_position,
                         cudaStream_t stream);

// ---------------------------------------------------------------------------
// SiLU-mul
// ---------------------------------------------------------------------------
Result<void> launch_silu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                             __nv_bfloat16* output, int64_t n, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Naive causal attention
// ---------------------------------------------------------------------------
Result<void> launch_naive_attention(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                             __nv_bfloat16* output, int num_tokens, int n_q_heads, int n_kv_heads,
                             int head_dim, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Elementwise helpers
// ---------------------------------------------------------------------------
Result<void> launch_element_add(__nv_bfloat16* dst, const __nv_bfloat16* src, int64_t n,
                                cudaStream_t stream);

Result<void> launch_split_qkv(const __nv_bfloat16* qkv, __nv_bfloat16* q, __nv_bfloat16* k,
                              __nv_bfloat16* v, int T, int nq, int nkv, int hd,
                              cudaStream_t stream);

// ---------------------------------------------------------------------------
// Embedding gather
// ---------------------------------------------------------------------------
Result<void> launch_embed(const __nv_bfloat16* embed_table, const int32_t* token_ids,
                           __nv_bfloat16* input_embeds, int num_tokens, int d_model,
                           cudaStream_t stream);

}  // namespace engine
}  // namespace ccinfer
