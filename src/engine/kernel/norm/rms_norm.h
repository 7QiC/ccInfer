#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ccinfer {
namespace engine {

void launch_rms_norm(const half* input, const half* weight, half* output, int rows, int dim,
                     float eps, cudaStream_t stream);

}  // namespace engine
}  // namespace ccinfer
