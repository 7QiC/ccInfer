#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ccinfer {
namespace engine {

class DeviceBackend;
class KVCacheManager;
template <typename DType>
class KVCacheStorage;

template <typename DType>
struct ExecutionContext {
    DeviceBackend* backend = nullptr;
    KVCacheStorage<DType>* kv_storage = nullptr;
    KVCacheManager* kv_mgr = nullptr;
    cudaStream_t stream = nullptr;
    int block_size = 16;
};

}  // namespace engine
}  // namespace ccinfer
