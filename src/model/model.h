#pragma once

#include "base/error.h"
#include "base/types.h"
#include "runtime/tensor.h"

namespace ccinfer {

struct ModelConfig;
class BlockStorage;
class Backend;
class StateStorage;

struct ForwardInput {
    // Exactly one of input_embeds / token_ids must be valid.
    Tensor input_embeds;  // [T, D] — pre-computed
    Tensor token_ids;     // [T] — device view, model embeds internally

    int num_tokens_ = 0;
    Tensor positions;          // [T]
    int max_position_id_ = 0;  // max position in batch; must be < rope cache capacity

    ForwardMode mode_ = ForwardMode::Prefill;

    // Paged-attention fields.
    Tensor slot_mapping;     // [num_tokens]
    Tensor block_table;      // [batch, max_blocks_per_req]
    Tensor query_start_loc;  // [batch + 1]
    Tensor context_lens;     // [batch]
    int batch_size_ = 0;
    int max_blocks_per_req_ = 0;

    // GDN state: state_mapping_[i] is the active state slot for batch item i
    // (-1 for Qwen3 or when no state is used). This is execution metadata only;
    // physical storage resources live in ModelExecutionContext.
    Tensor state_mapping;  // [batch_size], int32

    // CPU-side rows of the hidden tensor that need logits. The model computes
    // only these rows so the full [T, V] logits buffer is never materialized.
    std::vector<int32_t> logits_indices_host;
    int num_logits_ = 0;
};

// Runtime resources available to a model forward call. This is separate from
// ForwardInput: input describes how this batch maps to KV/state slots, ctx
// provides the storage/backend resources needed to execute those mappings.
struct ModelExecutionContext {
    Backend& backend;
    BlockStorage* block_storage;
    StateStorage* state_storage;

    ModelExecutionContext(Backend& backend, BlockStorage* block_storage = nullptr,
                          StateStorage* state_storage = nullptr)
        : backend(backend), block_storage(block_storage), state_storage(state_storage) {}
};

struct ForwardOutput {
    Tensor logits;      // [num_logits, vocab] float
    Tensor tokens_out;  // [batch_size] sampled token ids
};

class Model {
public:
    virtual ~Model() = default;

    virtual Result<void> forward(const ForwardInput& input, ForwardOutput& output,
                                 ModelExecutionContext& exec_ctx) = 0;

    virtual const ModelConfig& config() const = 0;
};

}  // namespace ccinfer
