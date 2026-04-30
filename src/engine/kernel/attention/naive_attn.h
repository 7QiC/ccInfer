#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/result.h"

namespace ccinfer {
namespace engine {

// Token-major naive causal attention.
//
// Layout:
//   q:      [num_tokens, n_q_heads,  head_dim]
//   k:      [num_tokens, n_kv_heads, head_dim]
//   v:      [num_tokens, n_kv_heads, head_dim]
//   output: [num_tokens, n_q_heads,  head_dim]
//
// GQA:
//   kv_head = q_head / (n_q_heads / n_kv_heads)
Result<void> naive_attention(const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                             __nv_bfloat16* output, int num_tokens, int n_q_heads, int n_kv_heads,
                             int head_dim, cudaStream_t stream);

}  // namespace engine
}  // namespace ccinfer
