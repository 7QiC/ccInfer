#include "engine/model/model_runner.h"

#include "engine/common/types.h"

namespace ccinfer {
namespace engine {

Result<std::vector<std::vector<int32_t>>> ModelRunner::execute(
    const PhysicalBatch& batch, const ExecutionContext& /*ctx*/) {
    // Placeholder — wired in Task 10 (real paged-attention forward).
    std::vector<std::vector<int32_t>> results;
    for (int i = 0; i < batch.batch_size; ++i) {
        results.push_back({42});  // dummy token
    }
    return results;
}

}  // namespace engine
}  // namespace ccinfer
