#pragma once

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ccinfer {
namespace engine {

void launch_silu_mul(const half* gate, const half* up, half* output, int64_t n,
                     cudaStream_t stream);

}  // namespace engine
}  // namespace ccinfer
