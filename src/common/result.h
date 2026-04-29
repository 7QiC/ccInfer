#pragma once
#include <expected>
#include "error_code.h"

template<typename T>
using Result = std::expected<T, ErrorCode>;
