#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>

#include "engine/common/dtype.h"

namespace ccinfer {
namespace engine {

// -----------------------------------------------------------------------------
// CUDA-free dtype tags.
//
// These tags are compile-time markers only.
// Backend-specific native types are mapped elsewhere, e.g.
//   engine/backend/cuda/cuda_dtype_traits.h
// -----------------------------------------------------------------------------

struct Float32Tag {};
struct Float16Tag {};
struct BFloat16Tag {};
struct Int8Tag {};

template <typename Tag>
struct DTypeTagTraits {
    static constexpr DType dtype = DType::kUnknown;
    static constexpr std::string_view name = "unknown";
};

template <>
struct DTypeTagTraits<Float32Tag> {
    static constexpr DType dtype = DType::kFloat32;
    static constexpr std::string_view name = "float32";
};

template <>
struct DTypeTagTraits<Float16Tag> {
    static constexpr DType dtype = DType::kFloat16;
    static constexpr std::string_view name = "float16";
};

template <>
struct DTypeTagTraits<BFloat16Tag> {
    static constexpr DType dtype = DType::kBFloat16;
    static constexpr std::string_view name = "bfloat16";
};

template <>
struct DTypeTagTraits<Int8Tag> {
    static constexpr DType dtype = DType::kInt8;
    static constexpr std::string_view name = "int8";
};

template <typename Tag>
inline constexpr DType dtype_tag_v = DTypeTagTraits<Tag>::dtype;

template <typename Tag>
inline constexpr std::string_view dtype_tag_name_v = DTypeTagTraits<Tag>::name;

// -----------------------------------------------------------------------------
// Quantization policy tags.
// CUDA-free. Backend-specific implementation lives elsewhere.
// -----------------------------------------------------------------------------

struct NoQuantPolicy {
    static constexpr bool is_quantized = false;
    static constexpr std::string_view name = "none";
};

struct Int8WeightOnlyPolicy {
    static constexpr bool is_quantized = true;
    static constexpr std::string_view name = "int8_weight_only";
};

struct Int4WeightOnlyPolicy {
    static constexpr bool is_quantized = true;
    static constexpr std::string_view name = "int4_weight_only";
};

// -----------------------------------------------------------------------------
// RunnerTraits.
//
// These are compile-time dtype profiles.
// They intentionally use dtype tags instead of CUDA native types.
// -----------------------------------------------------------------------------

template <typename WeightTagT, typename KVTagT, typename ActivationTagT, typename AccumTagT,
          typename LogitsTagT, typename QuantPolicyT = NoQuantPolicy>
struct RunnerTraits {
    using WeightTag = WeightTagT;
    using KVTag = KVTagT;
    using ActivationTag = ActivationTagT;
    using AccumTag = AccumTagT;
    using LogitsTag = LogitsTagT;
    using QuantPolicy = QuantPolicyT;

    static constexpr DType weight_dtype = dtype_tag_v<WeightTag>;
    static constexpr DType kv_dtype = dtype_tag_v<KVTag>;
    static constexpr DType activation_dtype = dtype_tag_v<ActivationTag>;
    static constexpr DType accum_dtype = dtype_tag_v<AccumTag>;
    static constexpr DType logits_dtype = dtype_tag_v<LogitsTag>;

    static constexpr bool is_quantized = QuantPolicy::is_quantized;
};

// -----------------------------------------------------------------------------
// Concrete CUDA-free profiles.
// -----------------------------------------------------------------------------

using BF16RunnerTraits = RunnerTraits<BFloat16Tag,  // weights
                                      BFloat16Tag,  // KV cache
                                      BFloat16Tag,  // activations
                                      Float32Tag,   // accumulation
                                      Float32Tag,   // logits
                                      NoQuantPolicy>;

using FP16RunnerTraits =
    RunnerTraits<Float16Tag, Float16Tag, Float16Tag, Float32Tag, Float32Tag, NoQuantPolicy>;

using FP16WeightBF16KVRunnerTraits =
    RunnerTraits<Float16Tag, BFloat16Tag, Float16Tag, Float32Tag, Float32Tag, NoQuantPolicy>;

using Int8WeightBF16KVRunnerTraits =
    RunnerTraits<Int8Tag, BFloat16Tag, BFloat16Tag, Float32Tag, Float32Tag, Int8WeightOnlyPolicy>;

// -----------------------------------------------------------------------------
// Helper predicates.
// -----------------------------------------------------------------------------

template <typename Traits>
inline constexpr bool runner_traits_valid_v =
    Traits::weight_dtype != DType::kUnknown && Traits::kv_dtype != DType::kUnknown &&
    Traits::activation_dtype != DType::kUnknown && Traits::accum_dtype != DType::kUnknown &&
    Traits::logits_dtype != DType::kUnknown;

template <typename Traits>
inline constexpr bool runner_uses_bf16_kv_v = std::is_same_v<typename Traits::KVTag, BFloat16Tag>;

template <typename Traits>
inline constexpr bool runner_uses_fp16_kv_v = std::is_same_v<typename Traits::KVTag, Float16Tag>;

}  // namespace engine
}  // namespace ccinfer
