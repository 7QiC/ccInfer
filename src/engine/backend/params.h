#pragma once

#include <cstdint>

namespace ccinfer {
namespace engine {

// -----------------------------------------------------------------------------
// GEMM
//
// Convention for current CUDA backend:
//
//   C = A @ B
//
// The exact row-major / column-major trick is handled inside CudaBackend::gemm.
// Keep this struct as a thin backend parameter carrier.
//
// For BF16 Phase 2:
//   a_, b_, c_ are expected to point to __nv_bfloat16 buffers.
// -----------------------------------------------------------------------------
struct GemmParams {
    const void* a_ = nullptr;
    const void* b_ = nullptr;
    void* c_ = nullptr;

    int m_ = 0;
    int n_ = 0;
    int k_ = 0;

    int lda_ = 0;
    int ldb_ = 0;
    int ldc_ = 0;

    bool trans_a_ = false;
    bool trans_b_ = false;

    // Optional stream override. If nullptr, backend default stream is used.
    void* stream_ = nullptr;
};

// -----------------------------------------------------------------------------
// RMSNorm
//
// Layout:
//   input:  [rows, dim]
//   weight: [dim]
//   output: [rows, dim]
//
// BF16 Phase 2:
//   input_, weight_, output_ point to __nv_bfloat16.
//   accumulation is FP32 inside CUDA kernel.
// -----------------------------------------------------------------------------
struct RmsNormParams {
    const void* input_ = nullptr;
    const void* weight_ = nullptr;
    void* output_ = nullptr;

    int rows_ = 0;
    int dim_ = 0;
    float eps_ = 1e-6f;

    void* stream_ = nullptr;
};

// -----------------------------------------------------------------------------
// RoPE
//
// Canonical activation layout:
//
//   q: [num_tokens, num_q_heads,  head_dim]
//   k: [num_tokens, num_kv_heads, head_dim]
//
// rope_cache:
//   [max_position, rotary_dim / 2]
//   each element is float2{cos, sin}
//
// BF16 Phase 2:
//   q_, k_ point to __nv_bfloat16.
// -----------------------------------------------------------------------------
struct RopeParams {
    void* q_ = nullptr;
    void* k_ = nullptr;

    const int32_t* positions_ = nullptr;
    const void* rope_cache_ = nullptr;

    int num_tokens_ = 0;
    int num_q_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int rotary_dim_ = 0;
    int max_position_ = 0;

    void* stream_ = nullptr;
};

// -----------------------------------------------------------------------------
// SiLU-mul
//
// Layout:
//   gate:   [n]
//   up:     [n]
//   output: [n]
//
// Computes:
//   output[i] = silu(gate[i]) * up[i]
//
// BF16 Phase 2:
//   gate_, up_, output_ point to __nv_bfloat16.
// -----------------------------------------------------------------------------
struct SiluMulParams {
    const void* gate_ = nullptr;
    const void* up_ = nullptr;
    void* output_ = nullptr;

    int64_t n_ = 0;

    void* stream_ = nullptr;
};

// -----------------------------------------------------------------------------
// Naive attention
//
// Phase 2 correctness baseline.
//
// Canonical token-major layout:
//
//   q:      [num_tokens, num_q_heads,  head_dim]
//   k:      [num_tokens, num_kv_heads, head_dim]
//   v:      [num_tokens, num_kv_heads, head_dim]
//   output: [num_tokens, num_q_heads,  head_dim]
//
// GQA mapping:
//   kv_head = q_head / (num_q_heads / num_kv_heads)
//
// BF16 Phase 2:
//   q_, k_, v_, output_ point to __nv_bfloat16.
//
// This is NOT the final high-performance attention backend.
// -----------------------------------------------------------------------------
struct NaiveAttnParams {
    const void* q_ = nullptr;
    const void* k_ = nullptr;
    const void* v_ = nullptr;
    void* output_ = nullptr;

    int num_tokens_ = 0;
    int num_q_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;

    void* stream_ = nullptr;
};

// -----------------------------------------------------------------------------
// Prefill attention
//
// Future high-performance prefill backend.
//
// Intended for FlashAttention-style / varlen prefill attention.
//
// Tentative layout:
//
//   q:      [total_query_tokens, num_q_heads,  head_dim]
//   output: [total_query_tokens, num_q_heads,  head_dim]
//
// K/V may come from either:
//   1. freshly computed k/v for current prefill tokens, or
//   2. KV cache blocks for prefix / previous context.
//
// These params are placeholders for Phase 3.
// -----------------------------------------------------------------------------
struct PrefillAttnParams {
    const void* q_ = nullptr;

    const void* k_cache_ = nullptr;
    const void* v_cache_ = nullptr;

    const int32_t* block_table_ = nullptr;
    const int32_t* slot_mapping_ = nullptr;
    const int32_t* query_start_loc_ = nullptr;
    const int32_t* seq_lens_ = nullptr;
    const int32_t* context_lens_ = nullptr;

    void* output_ = nullptr;

    int layer_ = 0;

    void* stream_ = nullptr;
};

// -----------------------------------------------------------------------------
// Decode attention
//
// Future PagedAttention decode backend.
//
// Tentative layout:
//
//   q:      [num_decode_tokens, num_q_heads, head_dim]
//   output: [num_decode_tokens, num_q_heads, head_dim]
//
// K/V are read from paged KV cache using block_table_ and context_lens_.
// -----------------------------------------------------------------------------
struct DecodeAttnParams {
    const void* q_ = nullptr;

    const void* k_cache_ = nullptr;
    const void* v_cache_ = nullptr;

    const int32_t* block_table_ = nullptr;
    const int32_t* context_lens_ = nullptr;

    void* output_ = nullptr;

    int layer_ = 0;

    void* stream_ = nullptr;
};

// -----------------------------------------------------------------------------
// Write KV cache
//
// Writes freshly computed K/V into paged KV cache.
//
// Tentative layout:
//
//   k_new: [total_tokens, num_kv_heads, head_dim]
//   v_new: [total_tokens, num_kv_heads, head_dim]
//
// slot_mapping_ maps each token to its physical KV-cache slot.
// -----------------------------------------------------------------------------
struct WriteKVCacheParams {
    const void* k_new_ = nullptr;
    const void* v_new_ = nullptr;

    void* k_cache_ = nullptr;
    void* v_cache_ = nullptr;

    const int32_t* slot_mapping_ = nullptr;

    int total_tokens_ = 0;

    void* stream_ = nullptr;
};

// -----------------------------------------------------------------------------
// Element-wise add (residual connection)
// -----------------------------------------------------------------------------
struct ElementAddParams {
    void* dst_ = nullptr;
    const void* src_ = nullptr;
    int64_t n_ = 0;
    void* stream_ = nullptr;
};

// -----------------------------------------------------------------------------
// Split fused QKV output into separate Q, K, V buffers
// -----------------------------------------------------------------------------
struct SplitQkvParams {
    const void* qkv_ = nullptr;
    void* q_ = nullptr;
    void* k_ = nullptr;
    void* v_ = nullptr;
    int num_tokens_ = 0;
    int num_q_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    void* stream_ = nullptr;
};

}  // namespace engine
}  // namespace ccinfer
