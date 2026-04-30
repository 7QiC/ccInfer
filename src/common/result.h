#pragma once

#include <expected>

#include "common/error_code.h"

namespace ccinfer {

template <typename T>
using Result = std::expected<T, ErrorCode>;

}  // namespace ccinfer
