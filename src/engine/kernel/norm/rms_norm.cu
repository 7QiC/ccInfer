#include "rms_norm.h"

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "backend/cuda/cuda_utils.h"

namespace ccinfer {
namespace engine {

namespace {

constexpr int kWarpSize = 32;

__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int kBlockSize>
__inline__ __device__ float block_reduce_sum(float val) {
    static_assert(kBlockSize % kWarpSize == 0);

    constexpr int kNumWarps = kBlockSize / kWarpSize;
    __shared__ float warp_sums[kNumWarps];

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        warp_sums[warp_id] = val;
    }

    __syncthreads();

    float block_sum = 0.0f;

    if (warp_id == 0) {
        block_sum = (lane < kNumWarps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);

        if (lane == 0) {
            warp_sums[0] = block_sum;
        }
    }

    __syncthreads();

    return warp_sums[0];
}

template <int kBlockSize>
__global__ void rms_norm_half2_kernel(const half* __restrict__ input,
                                      const half* __restrict__ weight, half* __restrict__ output,
                                      int rows, int dim, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;
    const int dim2 = dim / 2;

    const half* in_row = input + static_cast<int64_t>(row) * dim;
    half* out_row = output + static_cast<int64_t>(row) * dim;

    const half2* in2 = reinterpret_cast<const half2*>(in_row);
    const half2* w2 = reinterpret_cast<const half2*>(weight);
    half2* out2 = reinterpret_cast<half2*>(out_row);

    float sum_sq = 0.0f;

    for (int i = tid; i < dim2; i += kBlockSize) {
        half2 xh = in2[i];
        float2 x = __half22float2(xh);
        sum_sq += x.x * x.x + x.y * x.y;
    }

    sum_sq = block_reduce_sum<kBlockSize>(sum_sq);

    const float inv_rms = rsqrtf(sum_sq / static_cast<float>(dim) + eps);

    for (int i = tid; i < dim2; i += kBlockSize) {
        float2 x = __half22float2(in2[i]);
        float2 w = __half22float2(w2[i]);

        const float y0 = x.x * inv_rms * w.x;
        const float y1 = x.y * inv_rms * w.y;

        out2[i] = __floats2half2_rn(y0, y1);
    }
}

template <int kBlockSize>
__global__ void rms_norm_scalar_kernel(const half* __restrict__ input,
                                       const half* __restrict__ weight, half* __restrict__ output,
                                       int rows, int dim, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int tid = threadIdx.x;

    const half* in_row = input + static_cast<int64_t>(row) * dim;
    half* out_row = output + static_cast<int64_t>(row) * dim;

    float sum_sq = 0.0f;

    for (int i = tid; i < dim; i += kBlockSize) {
        float x = __half2float(in_row[i]);
        sum_sq += x * x;
    }

    sum_sq = block_reduce_sum<kBlockSize>(sum_sq);

    const float inv_rms = rsqrtf(sum_sq / static_cast<float>(dim) + eps);

    for (int i = tid; i < dim; i += kBlockSize) {
        float x = __half2float(in_row[i]);
        float w = __half2float(weight[i]);
        out_row[i] = __float2half_rn(x * inv_rms * w);
    }
}

inline bool is_aligned_4(const void* ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 0x3) == 0; }

}  // namespace

void launch_rms_norm(const half* input, const half* weight, half* output, int rows, int dim,
                     float eps, cudaStream_t stream) {
    constexpr int kBlockSize = 256;

    if (rows <= 0 || dim <= 0) {
        return;
    }

    dim3 grid(rows);
    dim3 block(kBlockSize);

    const bool can_use_half2 =
        (dim % 2 == 0) && is_aligned_4(input) && is_aligned_4(weight) && is_aligned_4(output);

    if (can_use_half2) {
        rms_norm_half2_kernel<kBlockSize>
            <<<grid, block, 0, stream>>>(input, weight, output, rows, dim, eps);
    } else {
        rms_norm_scalar_kernel<kBlockSize>
            <<<grid, block, 0, stream>>>(input, weight, output, rows, dim, eps);
    }

    CCINFER_CUDA_CHECK(cudaGetLastError());
}

}  // namespace engine
}  // namespace ccinfer
