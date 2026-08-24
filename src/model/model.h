#pragma once

#include "common/error_code.h"
#include "common/types.h"
#include "core/tensor.h"

namespace ccinfer {

struct ModelConfig;
class KVCacheManager;
class Backend;

struct ForwardInput {
    // Exactly one of input_embeds / token_ids must be valid.
    Tensor input_embeds;  // [T, D] — pre-computed
    Tensor token_ids;     // [T] — device view, model embeds internally

    int num_tokens_ = 0;
    Tensor positions;          // [T]
    int max_position_id_ = 0;  // max position in batch; must be < rope cache capacity

    ForwardMode mode_ = ForwardMode::Prefill;

    // Paged-attention fields — all mandatory when kv_mgr_ is set.
    KVCacheManager* kv_mgr_ = nullptr;
    Tensor slot_mapping;     // [num_tokens]
    Tensor block_table;      // [batch, max_blocks_per_req]
    Tensor query_start_loc;  // [batch + 1]
    Tensor context_lens;     // [batch]
    int batch_size_ = 0;
    int max_blocks_per_req_ = 0;
};

struct ForwardOutput {
    Tensor logits;            // [T, vocab] float
    Tensor tokens_out;        // [batch_size] sampled token ids
    Tensor eos_flags;         // [batch_size] future: per-seq EOS hit
    Tensor tokens_generated;  // [batch_size] future: speculative decode count
};

class Model {
public:
    virtual ~Model() = default;

    virtual Result<void> forward(const ForwardInput& input, ForwardOutput& output,
                                 Backend& backend) = 0;

    virtual const ModelConfig& config() const = 0;
};

}  // namespace ccinfer
