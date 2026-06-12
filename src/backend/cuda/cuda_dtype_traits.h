#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "core/traits.h"

namespace ccinfer {

template <typename Tag>
struct CudaNativeType;

template <>
struct CudaNativeType<Float32Tag> {
    using type = float;
};

template <>
struct CudaNativeType<Float16Tag> {
    using type = half;
};

template <>
struct CudaNativeType<BFloat16Tag> {
    using type = __nv_bfloat16;
};

template <>
struct CudaNativeType<Int8Tag> {
    using type = int8_t;
};

template <typename Tag>
using cuda_native_t = typename CudaNativeType<Tag>::type;

template <typename Traits>
struct CudaRunnerTypes {
    using WeightDType = cuda_native_t<typename Traits::WeightTag>;
    using KVDType = cuda_native_t<typename Traits::KVTag>;
    using ActivationDType = cuda_native_t<typename Traits::ActivationTag>;
    using AccumDType = cuda_native_t<typename Traits::AccumTag>;
    using LogitsDType = cuda_native_t<typename Traits::LogitsTag>;
};

}  // namespace ccinfer
