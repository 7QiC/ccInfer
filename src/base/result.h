#pragma once

#include <expected>

#include "base/error_code.h"

namespace ccinfer {

template <typename T>
using Result = std::expected<T, ErrorCode>;

}  // namespace ccinfer
