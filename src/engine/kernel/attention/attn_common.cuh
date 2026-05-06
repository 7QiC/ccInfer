#pragma once

#include <cstdint>

namespace ccinfer {
namespace engine {

__device__ __forceinline__ bool get_kv_cache_base_offset(
    int ctx_pos, int kv_head, int head_dim, int num_kv_heads, int cache_block_size,
    int max_blocks_per_req, const int32_t* __restrict__ req_blocks, int64_t* out_base_offset) {

    const int logical_block = ctx_pos / cache_block_size;
    const int offset_in_block = ctx_pos - logical_block * cache_block_size;

    if (logical_block < 0 || logical_block >= max_blocks_per_req) {
        return false;
    }

    const int physical_block = req_blocks[logical_block];
    if (physical_block < 0) {
        return false;
    }

    const int64_t slot =
        static_cast<int64_t>(physical_block) * cache_block_size + offset_in_block;
    *out_base_offset = (slot * num_kv_heads + kv_head) * static_cast<int64_t>(head_dim);
    return true;
}

}  // namespace engine
}  // namespace ccinfer
