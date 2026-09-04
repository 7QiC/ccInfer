#pragma once

// Third-party facade — the ONLY place the framework names the operator
// library (currently ccop). Framework code includes "facade/ops.h" and uses
// ccop:: symbols.
//
// Swapping the operator library means changing the includes and the namespace
// alias below (plus the submodule pointer / CMake link in src/CMakeLists.txt).
// If the replacement library has a different API shape, adapter forwarding
// functions go here too.

#include <ccop/device.h>
#include <ccop/dtype.h>
#include <ccop/error.h>
#include <ccop/execution_context.h>
#include <ccop/ops/causal_conv1d.h>
#include <ccop/ops/decode_attention.h>
#include <ccop/ops/element_add.h>
#include <ccop/ops/embed.h>
#include <ccop/ops/gated_delta_rule.h>
#include <ccop/ops/gdn_decay.h>
#include <ccop/ops/gemm.h>
#include <ccop/ops/greedy_sample.h>
#include <ccop/ops/l2_norm.h>
#include <ccop/ops/mul.h>
#include <ccop/ops/prefill_attention.h>
#include <ccop/ops/rms_norm.h>
#include <ccop/ops/rope.h>
#include <ccop/ops/sigmoid.h>
#include <ccop/ops/silu_mul.h>
#include <ccop/ops/split_qkv.h>
#include <ccop/ops/write_kv_cache.h>
#include <ccop/tensor.h>

#include "base/error.h"

namespace ccinfer {

// 换算子库时：替换上面的 include，并把下面这行改成新库的命名空间。
namespace ccop = ::ccop;

// Facade self-check: the replacement operator library must provide these core
// symbols; a failure here names the missing type directly.
static_assert(requires {
    typename ccop::DType;
    typename ccop::Device;
    typename ccop::ExecutionContext;
    typename ccop::Tensor;
    ccop::causal_conv1d;
    ccop::decode_attention;
    ccop::element_add;
    ccop::embed;
    ccop::gated_delta_rule;
    ccop::gdn_decay;
    ccop::gemm;
    ccop::greedy_sample;
    ccop::l2_norm;
    ccop::mul;
    ccop::prefill_attention;
    ccop::rms_norm;
    ccop::rope;
    ccop::sigmoid;
    ccop::silu_mul;
    ccop::split_qkv;
    ccop::write_kv_cache;
});
static_assert(sizeof(ccop::Device) == 8, "ccop::Device must stay an 8-byte value type");

// Map the operator library's backend-independent ErrorCode to ccinfer's
// ErrorCode. Framework call sites consume ccop Result through this facade so
// swapping the operator library only requires updating this mapping.
inline ErrorCode map_error(::ccop::ErrorCode code) noexcept {
    switch (code) {
        case ::ccop::ErrorCode::kOk:
            return ErrorCode::Ok;
        case ::ccop::ErrorCode::kInvalidArgument:
            return ErrorCode::InvalidArgument;
        case ::ccop::ErrorCode::kUnsupported:
            return ErrorCode::Unsupported;
        case ::ccop::ErrorCode::kOutOfMemory:
            return ErrorCode::CudaOutOfMemory;
        case ::ccop::ErrorCode::kRuntimeError:
            return ErrorCode::CudaRuntimeError;
    }
    return ErrorCode::InternalError;
}

inline Result<void> map_result(::ccop::Result<void> result) noexcept {
    if (!result) {
        return std::unexpected(map_error(result.error()));
    }
    return {};
}

}  // namespace ccinfer
