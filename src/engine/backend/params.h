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
// lda_, ldb_, ldc_ must be > 0.
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
};

// -----------------------------------------------------------------------------
// RMSNorm
//
// Layout:
//   input:  [rows, dim]
//   weight: [dim]
//   output: [rows, dim]
//
// eps_ must be > 0.
// -----------------------------------------------------------------------------
struct RmsNormParams {
    const void* input_ = nullptr;
    const void* weight_ = nullptr;
    void* output_ = nullptr;

    int rows_ = 0;
    int dim_ = 0;
    float eps_ = 1e-6f;
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
//   [rope_cache_max_position_, rotary_dim / 2]
//   each element is float2{cos, sin}
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
    int rope_cache_max_position_ = 0;  // RoPE cache capacity; valid pos < this
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
// -----------------------------------------------------------------------------
struct SiluMulParams {
    const void* gate_ = nullptr;
    const void* up_ = nullptr;
    void* output_ = nullptr;

    int64_t n_ = 0;
};

// -----------------------------------------------------------------------------
// Naive attention — single-sequence causal self-attention baseline.
//
// HF correctness reference.  Does NOT use KV cache; Q, K, V are all the same
// sequence length.  Causal mask: token at position i attends to [0, i].
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
};

// -----------------------------------------------------------------------------
// Prefill attention — FlashAttention-style tiling over paged KV cache.
//
// Q layout:  [total_tokens, num_q_heads, head_dim]
// Output:    [total_tokens, num_q_heads, head_dim]
//
// K/V are read exclusively from the KV cache (k_cache_/v_cache_).
// Caller must write new K/V into cache via write_kv_cache before calling.
//
// Causal semantics:
//   query_len_i  = query_start_loc_[i+1] - query_start_loc_[i]
//   prefix_len_i = context_lens_[i] - query_len_i
//   The j-th query token of request i may attend to [0, prefix_len_i + j].
//
// query_start_loc_[i] = start index in Q for request i (has batch_size_+1 elements)
// context_lens_[i]   = prefix_len + query_len = total KV length for request i
// block_table_: [batch_size_, max_blocks_per_req_] flattened, -1 for unused slots
struct PrefillAttnParams {
    const void* q_ = nullptr;
    const void* k_cache_ = nullptr;
    const void* v_cache_ = nullptr;

    const int32_t* block_table_ = nullptr;      // [batch, max_blocks_per_req]
    const int32_t* query_start_loc_ = nullptr;  // [batch + 1]
    const int32_t* context_lens_ = nullptr;     // [batch]

    void* output_ = nullptr;

    int num_tokens_ = 0;  // total query tokens = query_start_loc_[batch_size_]
    int batch_size_ = 0;
    int max_blocks_per_req_ = 0;
    int num_q_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int cache_block_size_ = 0;
};

// -----------------------------------------------------------------------------
// Decode attention — PagedAttention with online softmax.
//
// Q layout:  [batch_size, num_q_heads, head_dim]  (1 token per request)
// Output:    [batch_size, num_q_heads, head_dim]
//
// K/V read from paged KV cache via block_table_ and context_lens_.
// The current token's K/V must already be in cache via write_kv_cache.
// context_lens_[i] includes the current token.
struct DecodeAttnParams {
    const void* q_ = nullptr;
    const void* k_cache_ = nullptr;
    const void* v_cache_ = nullptr;

    const int32_t* block_table_ = nullptr;   // [batch, max_blocks_per_req]
    const int32_t* context_lens_ = nullptr;  // [batch]

    void* output_ = nullptr;

    int batch_size_ = 0;
    int max_blocks_per_req_ = 0;
    int num_q_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int cache_block_size_ = 0;
};

// Write KV cache — scatter newly computed K/V into paged KV cache.
//
//   k_new/v_new: [total_tokens, num_kv_heads, head_dim]
//   k_cache/v_cache: [max_slots, num_kv_heads, head_dim] per layer
//   slot_mapping_[t] = physical slot for token t
struct WriteKVCacheParams {
    const void* k_new_ = nullptr;
    const void* v_new_ = nullptr;
    void* k_cache_ = nullptr;
    void* v_cache_ = nullptr;
    const int32_t* slot_mapping_ = nullptr;

    int total_tokens_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int max_slots_ = 0;
};

// -----------------------------------------------------------------------------
// Element-wise add (residual connection)
// -----------------------------------------------------------------------------
struct ElementAddParams {
    void* dst_ = nullptr;
    const void* src_ = nullptr;
    int64_t n_ = 0;
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
};

// -----------------------------------------------------------------------------
// Embedding — GPU gather from embedding table
// -----------------------------------------------------------------------------
struct EmbedParams {
    const void* embed_table_ = nullptr;   // [vocab, d_model] bf16
    const int32_t* token_ids_ = nullptr;  // [num_tokens]
    void* input_embeds_ = nullptr;        // [num_tokens, d_model] bf16
    int num_tokens_ = 0;
    int d_model_ = 0;
};

// -----------------------------------------------------------------------------
// Sampling — backend-level.  Converted from common::SamplingParams.
//
// logits_:      [num_tokens, vocab] FP32
// logits_indices_: [batch_size]  — logits_[logits_indices_[i], :] is seq i's row
// tokens_out_:     [batch_size]
//
// Constraints: top_k_ in [0, vocab_size_], top_p_ in (0, 1], temperature_ >= 0.
// -----------------------------------------------------------------------------
struct SampleParams {
    const float* logits_ = nullptr;            // [num_tokens, vocab]
    const int32_t* logits_indices_ = nullptr;  // [batch_size]
    int32_t* tokens_out_ = nullptr;            // [batch_size]
    int num_tokens_ = 0;
    int batch_size_ = 0;
    int vocab_size_ = 0;
    int top_k_ = 0;
    float top_p_ = 1.0f;
    float temperature_ = 0.0f;
    uint32_t seed_ = 42;
};

}  // namespace engine
}  // namespace ccinfer
