#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "engine/backend/cuda/cuda_utils.h"
#include "engine/kernel/common.h"
#include "engine/kernel/cuda_common.cuh"
#include "engine/kernel/cuda_kernels.h"

namespace ccinfer {
namespace engine {

namespace {

// RMSNorm with BF16-matching accumulation.
//
// The sum-of-squares reduction rounds to BF16 after each element addition.
// This matches HuggingFace/PyTorch's BF16 reduction behaviour and produces
// logits within 0.07 max-diff of HF (vs 0.13 with FP32-only accumulation).
//
// The accumulator variable is float (FP32 register), but __float2bfloat16_rn
// after each addition discards lower mantissa bits, emulating BF16 storage.

template <int kBlockSize>
__global__ void rms_norm_bf162_kernel(const __nv_bfloat16* __restrict__ input,
                                      const __nv_bfloat16* __restrict__ weight,
                                      __nv_bfloat16* __restrict__ output, int rows, int dim,
                                      float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int dim2 = dim / 2;
    const __nv_bfloat16* in_row = input + static_cast<int64_t>(row) * dim;
    __nv_bfloat16* out_row = output + static_cast<int64_t>(row) * dim;

    const __nv_bfloat162* in2 = reinterpret_cast<const __nv_bfloat162*>(in_row);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out_row);

    // FP32 multiply + FP32 add, then round to BF16 after each element.
    // This produces the same rounding path as PyTorch's BF16 RMSNorm.
    float sum_sq = 0.0f;
    for (int i = tid; i < dim2; i += kBlockSize) {
        float2 x = __bfloat1622float2(in2[i]);
        float term = x.x * x.x + x.y * x.y;
        sum_sq = __bfloat162float(__float2bfloat16_rn(sum_sq + term));
    }
    sum_sq = block_reduce_sum<kBlockSize>(sum_sq);
    sum_sq = __bfloat162float(__float2bfloat16_rn(sum_sq));

    const float inv_rms = rsqrtf(sum_sq / static_cast<float>(dim) + eps);

    for (int i = tid; i < dim2; i += kBlockSize) {
        float2 x = __bfloat1622float2(in2[i]);
        float2 w = __bfloat1622float2(w2[i]);
        out2[i] = make_bfloat162(__float2bfloat16_rn(x.x * inv_rms * w.x),
                                 __float2bfloat16_rn(x.y * inv_rms * w.y));
    }
}

template <int kBlockSize>
__global__ void rms_norm_scalar_kernel(const __nv_bfloat16* __restrict__ input,
                                       const __nv_bfloat16* __restrict__ weight,
                                       __nv_bfloat16* __restrict__ output, int rows, int dim,
                                       float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const __nv_bfloat16* in_row = input + static_cast<int64_t>(row) * dim;
    __nv_bfloat16* out_row = output + static_cast<int64_t>(row) * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += kBlockSize) {
        float x = __bfloat162float(in_row[i]);
        float term = x * x;
        sum_sq = __bfloat162float(__float2bfloat16_rn(sum_sq + term));
    }
    sum_sq = block_reduce_sum<kBlockSize>(sum_sq);
    sum_sq = __bfloat162float(__float2bfloat16_rn(sum_sq));

    const float inv_rms = rsqrtf(sum_sq / static_cast<float>(dim) + eps);

    for (int i = tid; i < dim; i += kBlockSize) {
        float x = __bfloat162float(in_row[i]);
        float w = __bfloat162float(weight[i]);
        out_row[i] = __float2bfloat16_rn(x * inv_rms * w);
    }
}

}  // namespace

Result<void> launch_rms_norm(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                             __nv_bfloat16* output, int rows, int dim, float eps,
                             cudaStream_t stream) {
    constexpr int kBlockSize = 256;

    dim3 grid(rows);
    dim3 block(kBlockSize);

    const bool can_use_bf162 =
        (dim % 2 == 0) && is_aligned_4(input) && is_aligned_4(weight) && is_aligned_4(output);

    if (can_use_bf162)
        rms_norm_bf162_kernel<kBlockSize>
            <<<grid, block, 0, stream>>>(input, weight, output, rows, dim, eps);
    else
        rms_norm_scalar_kernel<kBlockSize>
            <<<grid, block, 0, stream>>>(input, weight, output, rows, dim, eps);

    return cuda_check_last_error();
}

}  // namespace engine
}  // namespace ccinfer
