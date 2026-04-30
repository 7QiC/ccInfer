#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/result.h"

namespace ccinfer {
namespace engine {

Result<void> launch_silu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                             __nv_bfloat16* output, int64_t n, cudaStream_t stream);

}  // namespace engine
}  // namespace ccinfer
