#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

#include "base/error_code.h"
#include "base/result.h"
#include "backend/cuda/cuda_utils.h"
#include "kernel/cuda_common.cuh"

namespace ccinfer {

namespace {

__global__ void greedy_sample_kernel(const float* logits, int32_t* tokens,
                                     const int32_t* logits_indices, int vocab_size, int batch_size,
                                     int num_tokens) {
    int b = blockIdx.x;
    if (b >= batch_size || vocab_size <= 0) return;

    int token_idx = logits_indices[b];
    if (token_idx < 0 || token_idx >= num_tokens) return;

    const float* seq_logits =
        logits + static_cast<size_t>(token_idx) * static_cast<size_t>(vocab_size);

    float best_val = -CUDART_INF_F;
    int32_t best_idx = 0;

    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
        float val = seq_logits[v];
        if (val > best_val || (val == best_val && v < best_idx)) {
            best_val = val;
            best_idx = v;
        }
    }

    extern __shared__ float shared_vals[];
    int32_t* shared_idxs = reinterpret_cast<int32_t*>(shared_vals + blockDim.x);

    shared_vals[threadIdx.x] = best_val;
    shared_idxs[threadIdx.x] = best_idx;
    __syncthreads();

    // kThreads must be power of two.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other_val = shared_vals[threadIdx.x + stride];
            int32_t other_idx = shared_idxs[threadIdx.x + stride];

            if (other_val > shared_vals[threadIdx.x] ||
                (other_val == shared_vals[threadIdx.x] && other_idx < shared_idxs[threadIdx.x])) {
                shared_vals[threadIdx.x] = other_val;
                shared_idxs[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        tokens[b] = shared_idxs[0];
    }
}

}  // namespace

Result<void> launch_greedy_sample(const float* logits, int32_t* tokens,
                                  const int32_t* logits_indices, int batch_size, int vocab_size,
                                  int num_tokens, cudaStream_t stream) {
    constexpr int kThreads = 256;
    static_assert((kThreads & (kThreads - 1)) == 0);

    size_t shared_bytes = static_cast<size_t>(kThreads) * (sizeof(float) + sizeof(int32_t));

    greedy_sample_kernel<<<batch_size, kThreads, shared_bytes, stream>>>(
        logits, tokens, logits_indices, vocab_size, batch_size, num_tokens);

    return cuda_check_last_error();
}

}  // namespace ccinfer
