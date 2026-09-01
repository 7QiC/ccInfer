#pragma once

#include "facade/ops.h"

namespace ccinfer {

// ExecutionTraits describes only runtime dense numerical dtypes. It deliberately
// does not describe weight storage (dense vs. quantized) or model topology.
struct ExecutionTraits {
    static constexpr ccop::DType activation_dtype = ccop::DType::kUnknown;
    static constexpr ccop::DType kv_dtype = ccop::DType::kUnknown;
    static constexpr ccop::DType accum_dtype = ccop::DType::kUnknown;
    static constexpr ccop::DType logits_dtype = ccop::DType::kUnknown;
    static constexpr ccop::DType recurrent_state_dtype = ccop::DType::kUnknown;
    static constexpr ccop::DType conv_state_dtype = ccop::DType::kUnknown;
};

struct Qwen3ExecutionTraits : public ExecutionTraits {
    static constexpr ccop::DType activation_dtype = ccop::DType::kBFloat16;
    static constexpr ccop::DType kv_dtype = ccop::DType::kBFloat16;
    static constexpr ccop::DType accum_dtype = ccop::DType::kFloat32;
    static constexpr ccop::DType logits_dtype = ccop::DType::kFloat32;
    static constexpr ccop::DType recurrent_state_dtype = ccop::DType::kFloat32;
    static constexpr ccop::DType conv_state_dtype = ccop::DType::kBFloat16;
};

struct Qwen3_5ExecutionTraits : public ExecutionTraits {
    static constexpr ccop::DType activation_dtype = ccop::DType::kBFloat16;
    static constexpr ccop::DType kv_dtype = ccop::DType::kBFloat16;
    static constexpr ccop::DType accum_dtype = ccop::DType::kFloat32;
    static constexpr ccop::DType logits_dtype = ccop::DType::kFloat32;
    static constexpr ccop::DType recurrent_state_dtype = ccop::DType::kFloat32;
    static constexpr ccop::DType conv_state_dtype = ccop::DType::kBFloat16;
};

template <typename Traits>
inline constexpr bool execution_traits_valid_v =
    Traits::activation_dtype != ccop::DType::kUnknown &&
    Traits::kv_dtype != ccop::DType::kUnknown &&
    Traits::accum_dtype != ccop::DType::kUnknown &&
    Traits::logits_dtype != ccop::DType::kUnknown &&
    Traits::recurrent_state_dtype != ccop::DType::kUnknown &&
    Traits::conv_state_dtype != ccop::DType::kUnknown;

}  // namespace ccinfer
