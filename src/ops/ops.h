#pragma once

// Operator library facade — the ONLY place the framework names the operator
// library (currently ccop). Framework code includes "ops/ops.h" and uses
// ccinfer::ops symbols.
//
// Swapping the operator library means changing the includes and the namespace
// alias below (plus the submodule pointer / CMake link in src/CMakeLists.txt).
// If the replacement library has a different API shape, adapter forwarding
// functions go here too.

#include <ccop/device.h>
#include <ccop/dtype.h>
#include <ccop/error.h>
#include <ccop/execution_context.h>
#include <ccop/ops/decode_attention.h>
#include <ccop/ops/element_add.h>
#include <ccop/ops/embed.h>
#include <ccop/ops/gemm.h>
#include <ccop/ops/greedy_sample.h>
#include <ccop/ops/naive_attention.h>
#include <ccop/ops/prefill_attention.h>
#include <ccop/ops/reduce.h>
#include <ccop/ops/rms_norm.h>
#include <ccop/ops/rope.h>
#include <ccop/ops/silu_mul.h>
#include <ccop/ops/softmax.h>
#include <ccop/ops/split_qkv.h>
#include <ccop/ops/write_kv_cache.h>
#include <ccop/tensor.h>

#include "base/error_code.h"
#include "base/result.h"

namespace ccinfer::ops {

// 换算子库时：替换上面的 include，并把下面这行改成新库的命名空间。
using namespace ::ccop;

// Facade self-check: the replacement operator library must provide these core
// symbols; a failure here names the missing type directly.
static_assert(requires {
    typename ops::DType;
    typename ops::Device;
    typename ops::ExecutionContext;
    typename ops::Tensor;
    ops::decode_attention;
    ops::element_add;
    ops::embed;
    ops::gemm;
    ops::greedy_sample;
    ops::naive_attention;
    ops::prefill_attention;
    ops::reduce_sum_rows;
    ops::rms_norm;
    ops::rope;
    ops::silu_mul;
    ops::softmax;
    ops::split_qkv;
    ops::write_kv_cache;
});
static_assert(sizeof(ops::Device) == 8, "ops::Device must stay an 8-byte value type");

// Map the operator library's backend-independent ErrorCode to ccinfer's
// ErrorCode. Framework call sites consume ccop Result through this facade so
// swapping the operator library only requires updating this mapping.
inline ccinfer::ErrorCode map_error(::ccop::ErrorCode code) noexcept {
    switch (code) {
        case ::ccop::ErrorCode::kOk:
            return ccinfer::ErrorCode::Ok;
        case ::ccop::ErrorCode::kInvalidArgument:
            return ccinfer::ErrorCode::InvalidArgument;
        case ::ccop::ErrorCode::kUnsupported:
            return ccinfer::ErrorCode::Unsupported;
        case ::ccop::ErrorCode::kOutOfMemory:
            return ccinfer::ErrorCode::CudaOutOfMemory;
        case ::ccop::ErrorCode::kRuntimeError:
            return ccinfer::ErrorCode::CudaRuntimeError;
    }
    return ccinfer::ErrorCode::InternalError;
}

inline ccinfer::Result<void> map_result(::ccop::Result<void> result) noexcept {
    if (!result) {
        return std::unexpected(map_error(result.error()));
    }
    return {};
}

}  // namespace ccinfer::ops
