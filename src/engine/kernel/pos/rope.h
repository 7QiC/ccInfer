#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/result.h"

namespace ccinfer {
namespace engine {

Result<void> launch_rope(__nv_bfloat16* q, __nv_bfloat16* k, const int32_t* positions,
                         const float2* rope_cache, int num_tokens, int num_q_heads,
                         int num_kv_heads, int head_dim, int rotary_dim, int max_position,
                         cudaStream_t stream);

}  // namespace engine
}  // namespace ccinfer
