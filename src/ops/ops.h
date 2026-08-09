#pragma once

// -----------------------------------------------------------------------------
// Operator library facade — the ONLY place the framework names the operator
// library (currently ccop). Framework code includes "ops/ops.h" and uses
// ccinfer::ops symbols.
//
// Swapping the operator library means changing the includes and the namespace
// alias below (plus the submodule pointer / CMake link in src/CMakeLists.txt).
// If the replacement library has a different API shape, adapter forwarding
// functions go here too.
// -----------------------------------------------------------------------------

#include <ccop/device.h>
#include <ccop/dtype.h>
#include <ccop/execution_context.h>
#include <ccop/tensor.h>

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
});
static_assert(sizeof(ops::Device) == 8, "ops::Device must stay an 8-byte value type");

}  // namespace ccinfer::ops
