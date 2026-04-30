#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ccinfer {
namespace engine {

void launch_rope(half* q, half* k, const int32_t* positions, const float2* rope_cache,
                 int num_tokens, int num_q_heads, int num_kv_heads, int head_dim, int rotary_dim,
                 int max_position, cudaStream_t stream);

}  // namespace engine
}  // namespace ccinfer
