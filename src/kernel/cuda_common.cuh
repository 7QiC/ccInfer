#pragma once

#include <cfloat>
#include <cstddef>

namespace ccinfer {

__host__ __device__ constexpr std::size_t align_up(std::size_t x, std::size_t align) {
    return (x + align - 1) & ~(align - 1);
}

constexpr int kWarpSize = 32;

__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <int kBlockSize>
__inline__ __device__ float block_reduce_sum(float val) {
    static_assert(kBlockSize % kWarpSize == 0, "block size must be multiple of warp size");

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
__inline__ __device__ float block_reduce_max(float val) {
    static_assert(kBlockSize % kWarpSize == 0, "block size must be multiple of warp size");

    constexpr int kNumWarps = kBlockSize / kWarpSize;
    __shared__ float warp_max[kNumWarps];

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;

    val = warp_reduce_max(val);

    if (lane == 0) {
        warp_max[warp_id] = val;
    }

    __syncthreads();

    float block_max = -FLT_MAX;

    if (warp_id == 0) {
        block_max = (lane < kNumWarps) ? warp_max[lane] : -FLT_MAX;
        block_max = warp_reduce_max(block_max);

        if (lane == 0) {
            warp_max[0] = block_max;
        }
    }

    __syncthreads();

    return warp_max[0];
}

}  // namespace ccinfer
