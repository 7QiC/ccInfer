#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "base/result.h"
#include "backend/default_backend.h"

namespace ccinfer {

class DeviceBuffer;

class KVCacheStorage {
public:
    KVCacheStorage() = default;
    ~KVCacheStorage();
    KVCacheStorage(KVCacheStorage&&) noexcept;
    KVCacheStorage& operator=(KVCacheStorage&&) noexcept;

    KVCacheStorage(const KVCacheStorage&) = delete;
    KVCacheStorage& operator=(const KVCacheStorage&) = delete;

    template <typename KVDType>
    static Result<std::unique_ptr<KVCacheStorage>> create(DefaultBackend& backend, int num_layers,
                                                          int max_blocks, int block_size,
                                                          int num_kv_heads, int head_dim);

    void* k_data();
    void* v_data();
    const void* k_data() const;
    const void* v_data() const;

    void* k_layer(int layer);
    void* v_layer(int layer);
    const void* k_layer(int layer) const;
    const void* v_layer(int layer) const;

    int num_layers() const { return num_layers_; }
    int max_slots() const { return max_slots_; }
    int64_t layer_stride() const { return layer_stride_; }
    std::size_t elem_size() const { return elem_size_; }

private:
    Result<void> init(DefaultBackend& backend, int num_layers, int max_blocks, int block_size,
                      int num_kv_heads, int head_dim, std::size_t elem_size);

    std::unique_ptr<DeviceBuffer> k_data_;
    std::unique_ptr<DeviceBuffer> v_data_;
    int64_t layer_stride_ = 0;
    std::size_t elem_size_ = 0;
    int max_slots_ = 0;
    int num_layers_ = 0;
};

template <typename KVDType>
Result<std::unique_ptr<KVCacheStorage>> KVCacheStorage::create(DefaultBackend& backend,
                                                               int num_layers, int max_blocks,
                                                               int block_size, int num_kv_heads,
                                                               int head_dim) {
    auto storage = std::make_unique<KVCacheStorage>();
    auto r = storage->init(backend, num_layers, max_blocks, block_size, num_kv_heads, head_dim,
                           sizeof(KVDType));
    if (!r) return std::unexpected(r.error());
    return storage;
}

}  // namespace ccinfer
