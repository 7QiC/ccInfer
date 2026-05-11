#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "engine/backend/cuda/cuda_utils.h"
#include "engine/kernel/common.h"
#include "engine/kernel/cuda_kernels.h"

namespace ccinfer {
namespace engine {

namespace {

__global__ void silu_mul_bf162_kernel(const __nv_bfloat16* __restrict__ gate,
                                      const __nv_bfloat16* __restrict__ up,
                                      __nv_bfloat16* __restrict__ output, int64_t n) {
    int64_t idx2 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n2 = n / 2;
    if (idx2 < n2) {
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(gate);
        const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(up);
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(output);

        float2 g = __bfloat1622float2(__ldg(g2 + idx2));
        float2 u = __bfloat1622float2(__ldg(u2 + idx2));
        float s0 = g.x / (1.0f + expf(-g.x)) * u.x;
        float s1 = g.y / (1.0f + expf(-g.y)) * u.y;
        o2[idx2] = make_bfloat162(__float2bfloat16_rn(s0), __float2bfloat16_rn(s1));
    }
    if ((n & 1) && idx2 == n2) {
        int64_t idx = n - 1;
        float g = __bfloat162float(gate[idx]);
        float u = __bfloat162float(up[idx]);
        output[idx] = __float2bfloat16_rn(g / (1.0f + expf(-g)) * u);
    }
}

__global__ void silu_mul_scalar_kernel(const __nv_bfloat16* __restrict__ gate,
                                       const __nv_bfloat16* __restrict__ up,
                                       __nv_bfloat16* __restrict__ output, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    output[idx] = __float2bfloat16_rn(g / (1.0f + expf(-g)) * u);
}

}  // namespace

Result<void> launch_silu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                             __nv_bfloat16* output, int64_t n, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    bool use_bf162 = is_aligned_4(gate) && is_aligned_4(up) && is_aligned_4(output);

    if (use_bf162) {
        int64_t work = (n + 1) / 2;
        silu_mul_bf162_kernel<<<ceil_div_i64(work, kBlockSize), kBlockSize, 0, stream>>>(gate, up,
                                                                                         output, n);
    } else {
        silu_mul_scalar_kernel<<<ceil_div_i64(n, kBlockSize), kBlockSize, 0, stream>>>(gate, up,
                                                                                       output, n);
    }
    return cuda_check_last_error();
}

}  // namespace engine
}  // namespace ccinfer
