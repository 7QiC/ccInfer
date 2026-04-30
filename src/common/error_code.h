#pragma once

#include <cstdint>
#include <string_view>

namespace ccinfer {

enum class ErrorCode : uint16_t {
    Ok = 0,

    CudaOutOfMemory,
    CudaLaunchFailed,
    CudaInvalidValue,
    CudaRuntimeError,
    CublasError,

    InvalidArgument,

    ModelLoadFailed,
    ModelConfigInvalid,
    ModelShapeMismatch,
    ModelUnsupportedArch,
    ModelUnsupportedDType,

    RequestTooLong,
    RequestCancelled,
    RequestTimeout,
    ServerShuttingDown,
};

inline constexpr std::string_view error_message(ErrorCode c) noexcept {
    switch (c) {
        case ErrorCode::Ok:
            return "ok";

        case ErrorCode::CudaOutOfMemory:
            return "CUDA out of memory";
        case ErrorCode::CudaLaunchFailed:
            return "CUDA kernel launch failed";
        case ErrorCode::CudaInvalidValue:
            return "CUDA invalid value";
        case ErrorCode::CudaRuntimeError:
            return "CUDA runtime error";
        case ErrorCode::CublasError:
            return "cuBLAS error";

        case ErrorCode::InvalidArgument:
            return "invalid argument";

        case ErrorCode::ModelLoadFailed:
            return "model load failed";
        case ErrorCode::ModelConfigInvalid:
            return "model config invalid";
        case ErrorCode::ModelShapeMismatch:
            return "model shape mismatch";
        case ErrorCode::ModelUnsupportedArch:
            return "unsupported model architecture";
        case ErrorCode::ModelUnsupportedDType:
            return "unsupported model dtype";

        case ErrorCode::RequestTooLong:
            return "request too long";
        case ErrorCode::RequestCancelled:
            return "request cancelled";
        case ErrorCode::RequestTimeout:
            return "request timed out";
        case ErrorCode::ServerShuttingDown:
            return "server shutting down";
    }

    return "unknown error";
}

}  // namespace ccinfer
