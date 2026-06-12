#pragma once

#include <cstdint>

#include "base/result.h"
#include "base/types.h"
#include "backend/params.h"
#include "backend/default_backend.h"

namespace ccinfer {

struct ModelConfig;
class KVCacheManager;

struct ForwardInput {
    // Exactly one of input_embeds_ / token_ids_ must be non-null.
    const void* input_embeds_ = nullptr;  // [T, D] — pre-computed
    const int32_t* token_ids_ = nullptr;  // [T] — device pointer, model embeds internally

    int num_tokens_ = 0;
    const int32_t* positions_ = nullptr;  // [T] device pointer
    int max_position_id_ = 0;             // max position in batch; must be < rope cache capacity

    ForwardMode mode_ = ForwardMode::Prefill;

    // Paged-attention fields — all mandatory when kv_mgr_ is set.
    KVCacheManager* kv_mgr_ = nullptr;
    const int32_t* slot_mapping_ = nullptr;     // [num_tokens]
    const int32_t* block_table_ = nullptr;      // [batch, max_blocks_per_req]
    const int32_t* query_start_loc_ = nullptr;  // [batch + 1]
    const int32_t* context_lens_ = nullptr;     // [batch]
    int batch_size_ = 0;
    int max_blocks_per_req_ = 0;
};

struct ForwardOutput {
    void* logits_ = nullptr;           // [T, vocab] float
    int32_t* tokens_out_ = nullptr;    // [batch_size] sampled token ids
    int32_t* eos_flags_ = nullptr;     // [batch_size] future: per-seq EOS hit
    int* tokens_generated_ = nullptr;  // [batch_size] future: speculative decode count
};

class Model {
public:
    virtual ~Model() = default;

    virtual Result<void> forward(const ForwardInput& input, ForwardOutput& output,
                                 DefaultBackend& backend) = 0;

    virtual const ModelConfig& config() const = 0;
    virtual const char* architecture() const = 0;
};

}  // namespace ccinfer
