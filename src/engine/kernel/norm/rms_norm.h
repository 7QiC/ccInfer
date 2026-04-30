#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/result.h"

namespace ccinfer {
namespace engine {

Result<void> launch_rms_norm(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                             __nv_bfloat16* output, int rows, int dim, float eps,
                             cudaStream_t stream);

}  // namespace engine
}  // namespace ccinfer
