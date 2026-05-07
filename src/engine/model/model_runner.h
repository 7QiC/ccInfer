#pragma once

#include <cstdint>
#include <vector>

#include "common/result.h"

namespace ccinfer {
namespace engine {

class Model;
struct PhysicalBatch;
class DeviceBackend;
template <typename DType>
class KVCacheStorage;

class ModelRunner {
public:
    template <typename Traits, typename KVDType>
    static Result<std::vector<std::vector<int32_t>>> execute(Model& model,
                                                             const PhysicalBatch& batch,
                                                             DeviceBackend& backend,
                                                             KVCacheStorage<KVDType>& kv_storage);
};

}  // namespace engine
}  // namespace ccinfer

#include "engine/model/model_runner.tpp"