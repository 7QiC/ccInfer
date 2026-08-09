#pragma once

#include <string_view>
#include <type_traits>

#include "ops/ops.h"

namespace ccinfer {

// -----------------------------------------------------------------------------
// DType profile: compile-time dtype configuration for a model run.
//
// dtype tags come from ccop (single source of truth). CUDA native types are
// obtained via ops::native_t<Tag> after including <ccop/cuda/dtype_cuda.h>
// inside CUDA translation units.
// -----------------------------------------------------------------------------

template <typename WeightTagT, typename KVTagT, typename ActivationTagT, typename AccumTagT,
          typename LogitsTagT>
struct DTypeProfile {
    using WeightTag = WeightTagT;
    using KVTag = KVTagT;
    using ActivationTag = ActivationTagT;
    using AccumTag = AccumTagT;
    using LogitsTag = LogitsTagT;

    static constexpr ops::DType weight_dtype = ops::dtype_v<WeightTag>;
    static constexpr ops::DType kv_dtype = ops::dtype_v<KVTag>;
    static constexpr ops::DType activation_dtype = ops::dtype_v<ActivationTag>;
    static constexpr ops::DType accum_dtype = ops::dtype_v<AccumTag>;
    static constexpr ops::DType logits_dtype = ops::dtype_v<LogitsTag>;
};

// Concrete dtype profiles.
using BF16DTypeProfile =
    DTypeProfile<ops::BFloat16Tag,  // weights
                 ops::BFloat16Tag,  // KV cache
                 ops::BFloat16Tag,  // activations
                 ops::Float32Tag,   // accumulation
                 ops::Float32Tag>;  // logits

using FP16DTypeProfile = DTypeProfile<ops::Float16Tag, ops::Float16Tag, ops::Float16Tag,
                                      ops::Float32Tag, ops::Float32Tag>;

using FP16WeightBF16KVDTypeProfile =
    DTypeProfile<ops::Float16Tag, ops::BFloat16Tag, ops::Float16Tag, ops::Float32Tag,
                 ops::Float32Tag>;

using Int8WeightBF16KVDTypeProfile =
    DTypeProfile<ops::Int8Tag, ops::BFloat16Tag, ops::BFloat16Tag, ops::Float32Tag,
                 ops::Float32Tag>;

// -----------------------------------------------------------------------------
// Quantization policy tags.
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
// RunnerTraits = dtype profile + quantization policy.
// -----------------------------------------------------------------------------

template <typename ProfileT, typename QuantPolicyT = NoQuantPolicy>
struct RunnerTraits {
    using Profile = ProfileT;
    using QuantPolicy = QuantPolicyT;

    using WeightTag = typename ProfileT::WeightTag;
    using KVTag = typename ProfileT::KVTag;
    using ActivationTag = typename ProfileT::ActivationTag;
    using AccumTag = typename ProfileT::AccumTag;
    using LogitsTag = typename ProfileT::LogitsTag;

    static constexpr bool is_quantized = QuantPolicyT::is_quantized;
};

using BF16RunnerTraits = RunnerTraits<BF16DTypeProfile>;
using FP16RunnerTraits = RunnerTraits<FP16DTypeProfile>;
using FP16WeightBF16KVRunnerTraits = RunnerTraits<FP16WeightBF16KVDTypeProfile>;
using Int8WeightBF16KVRunnerTraits =
    RunnerTraits<Int8WeightBF16KVDTypeProfile, Int8WeightOnlyPolicy>;

// -----------------------------------------------------------------------------
// Helper predicates.
// -----------------------------------------------------------------------------

template <typename Traits>
inline constexpr bool runner_traits_valid_v =
    Traits::Profile::weight_dtype != ops::DType::kUnknown &&
    Traits::Profile::kv_dtype != ops::DType::kUnknown &&
    Traits::Profile::activation_dtype != ops::DType::kUnknown &&
    Traits::Profile::accum_dtype != ops::DType::kUnknown &&
    Traits::Profile::logits_dtype != ops::DType::kUnknown;

}  // namespace ccinfer
