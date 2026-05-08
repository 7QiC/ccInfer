#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

#include "engine/backend/backend.h"
#include "engine/cache/kv_cache_storage.h"
#include "engine/common/types.h"
#include "engine/model/model.h"

namespace ccinfer {
namespace engine {

template <typename Backend, typename Model, typename Traits, typename KVDType>
Result<std::vector<std::vector<int32_t>>> ModelRunner::execute(
    Model& model, const PhysicalBatch& batch, Backend& backend,
    KVCacheStorage<KVDType>& kv_storage) {
    // Placeholder implementation.
    //
    // Real implementation will:
    //   1. read token_ids / positions / slot_mapping from PhysicalBatch
    //   2. run embedding
    //   3. run transformer layers with paged attention
    //   4. write/read KV cache through kv_storage
    //   5. run lm_head
    //   6. sample one token for each sequence that requires sampling
    //
    // For now, return one dummy token per physical sequence so that
    // Engine -> Worker -> ModelRunner -> Scheduler -> SSE can be wired end-to-end.

    (void)model;
    (void)backend;
    (void)kv_storage;

    assert(batch.batch_size >= 0);
    assert(batch.num_tokens >= 0);

    std::vector<std::vector<int32_t>> results;
    results.reserve(static_cast<size_t>(batch.batch_size));

    for (int i = 0; i < batch.batch_size; ++i) {
        // Dummy token. Later this should be the sampled token from lm_head.
        results.push_back({42});
    }

    return results;
}

}  // namespace engine
}  // namespace ccinfer
