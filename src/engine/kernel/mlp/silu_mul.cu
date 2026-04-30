#include "engine/kernel/mlp/silu_mul.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "engine/backend/cuda/cuda_utils.h"

namespace ccinfer {
namespace engine {

namespace {

inline bool is_aligned_4(const void* ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 0x3u) == 0; }

inline int ceil_div_int64_to_int(int64_t a, int b) { return static_cast<int>((a + b - 1) / b); }

// output[i] = gate[i] * sigmoid(gate[i]) * up[i]
__global__ void silu_mul_half2_kernel(const half* __restrict__ gate, const half* __restrict__ up,
                                      half* __restrict__ output, int64_t n) {
    const int64_t idx2 = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t n2 = n / 2;

    if (idx2 < n2) {
        const half2* gate2 = reinterpret_cast<const half2*>(gate);
        const half2* up2 = reinterpret_cast<const half2*>(up);
        half2* out2 = reinterpret_cast<half2*>(output);

        const half2 g_h = __ldg(gate2 + idx2);
        const half2 u_h = __ldg(up2 + idx2);

        const float2 g = __half22float2(g_h);
        const float2 u = __half22float2(u_h);

        const float silu0 = g.x / (1.0f + expf(-g.x));
        const float silu1 = g.y / (1.0f + expf(-g.y));

        out2[idx2] = __floats2half2_rn(silu0 * u.x, silu1 * u.y);
    }

    // Handle odd tail element.
    if ((n & 1) && idx2 == n2) {
        const int64_t idx = n - 1;

        const float g = __half2float(gate[idx]);
        const float u = __half2float(up[idx]);
        const float silu = g / (1.0f + expf(-g));

        output[idx] = __float2half_rn(silu * u);
    }
}

__global__ void silu_mul_scalar_kernel(const half* __restrict__ gate, const half* __restrict__ up,
                                       half* __restrict__ output, int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    const float g = __half2float(gate[idx]);
    const float u = __half2float(up[idx]);

    const float silu = g / (1.0f + expf(-g));
    output[idx] = __float2half_rn(silu * u);
}

}  // namespace

void launch_silu_mul(const half* gate, const half* up, half* output, int64_t n,
                     cudaStream_t stream) {
    if (n <= 0) {
        return;
    }

    if (gate == nullptr || up == nullptr || output == nullptr) {
        return;
    }

    constexpr int kBlockSize = 256;

    const bool can_use_half2 = is_aligned_4(gate) && is_aligned_4(up) && is_aligned_4(output);

    if (can_use_half2) {
        const int64_t work_items = (n + 1) / 2;
        const int grid = ceil_div_int64_to_int(work_items, kBlockSize);

        silu_mul_half2_kernel<<<grid, kBlockSize, 0, stream>>>(gate, up, output, n);
    } else {
        const int grid = ceil_div_int64_to_int(n, kBlockSize);

        silu_mul_scalar_kernel<<<grid, kBlockSize, 0, stream>>>(gate, up, output, n);
    }

    CCINFER_CUDA_CHECK(cudaGetLastError());
}

}  // namespace engine
}  // namespace ccinfer
