#pragma once

#include <cstdint>

namespace ccinfer {
namespace engine {

struct GemmParams {
    const void* A;
    const void* B;
    void* C;
    int m, n, k;        // dimensions
    int lda, ldb, ldc;  // leading dimensions
    bool trans_a = false;
    bool trans_b = false;
    void* stream = nullptr;
};

struct RmsNormParams {
    const void* input;
    const void* weight;
    void* output;
    int rows;
    int dim;
    float eps;
    void* stream = nullptr;
};

struct RopeParams {
    void* q;
    void* k;
    const int32_t* positions;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    float rope_theta;
    void* stream = nullptr;
};

struct SiluMulParams {
    const void* gate;
    const void* up;
    void* output;
    int64_t n;
    void* stream = nullptr;
};

struct PrefillAttnParams {
    const void* Q;
    const void* K_cache;
    const void* V_cache;
    const int32_t* block_table;
    const int32_t* slot_mapping;
    const int32_t* query_start_loc;
    const int32_t* seq_lens;
    const int32_t* context_lens;
    void* output;
    int layer;
    void* stream = nullptr;
};

struct DecodeAttnParams {
    const void* Q;
    const void* K_cache;
    const void* V_cache;
    const int32_t* block_table;
    const int32_t* context_lens;
    void* output;
    int layer;
    void* stream = nullptr;
};

struct WriteKVCacheParams {
    const void* K_new;
    const void* V_new;
    void* K_cache;
    void* V_cache;
    const int32_t* slot_mapping;
    int total_tokens;
    void* stream = nullptr;
};

}  // namespace engine
}  // namespace ccinfer
