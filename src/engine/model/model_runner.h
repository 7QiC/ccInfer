#pragma once

#include <cstdint>
#include <vector>

#include "common/result.h"

namespace ccinfer {
namespace engine {

struct PhysicalBatch;
struct ExecutionContext;

class ModelRunner {
public:
    // Stateless per-invocation. Model and backend come from ExecutionContext.
    static Result<std::vector<std::vector<int32_t>>> execute(const PhysicalBatch& batch,
                                                             const ExecutionContext& ctx);
};

}  // namespace engine
}  // namespace ccinfer
