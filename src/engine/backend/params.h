#pragma once

#include <cstdint>

#include "cuda_runtime.h"

namespace ccinfer {
namespace engine {

struct GemmParams {
    const void* a_;
    const void* b_;
    void* c_;
    int m_ = 0, n_ = 0, k_ = 0;
    int lda_ = 0, ldb_ = 0, ldc_ = 0;
    bool trans_a_ = false;
    bool trans_b_ = false;
    void* stream_ = nullptr;
};

struct RmsNormParams {
    const void* input_;
    const void* weight_;
    void* output_;
    int rows_ = 0;
    int dim_ = 0;
    float eps_ = 0.0f;
    void* stream_ = nullptr;
};

struct RopeParams {
    void* q_;
    void* k_;
    const int32_t* positions_;
    const float2* rope_cache_;
    int num_tokens_ = 0;
    int num_q_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int rotary_dim_ = 0;
    int max_position_ = 0;
    void* stream_ = nullptr;
};

struct SiluMulParams {
    const void* gate_;
    const void* up_;
    void* output_;
    int64_t n_ = 0;
    void* stream_ = nullptr;
};

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

struct WriteKVCacheParams {
    const void* k_new_ = nullptr;
    const void* v_new_ = nullptr;
    void* k_cache_ = nullptr;
    void* v_cache_ = nullptr;
    const int32_t* slot_mapping_ = nullptr;
    int total_tokens_ = 0;
    void* stream_ = nullptr;
};

}  // namespace engine
}  // namespace ccinfer
