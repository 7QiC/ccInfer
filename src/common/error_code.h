#pragma once
#include <cstdint>
#include <string_view>

namespace ccinfer {

enum class ErrorCode : uint8_t {
    Ok = 0,
    CudaOutOfMemory,
    CudaLaunchFailed,
    CudaInvalidValue,
    KVBlockExhausted,
    KVBlockDoubleFree,
    ModelLoadFailed,
    ModelShapeMismatch,
    ModelUnsupportedArch,
    RequestTooLong,
    RequestCancelled,
    RequestTimeout,
    ServerShuttingDown,
};

inline std::string_view error_message(ErrorCode c) {
    switch (c) {
        case ErrorCode::Ok:              return "ok";
        case ErrorCode::CudaOutOfMemory:  return "CUDA out of memory";
        case ErrorCode::CudaLaunchFailed: return "CUDA kernel launch failed";
        case ErrorCode::CudaInvalidValue: return "CUDA invalid value";
        case ErrorCode::KVBlockExhausted: return "KV cache blocks exhausted";
        case ErrorCode::KVBlockDoubleFree:return "KV cache block double free";
        case ErrorCode::ModelLoadFailed:  return "model load failed";
        case ErrorCode::ModelShapeMismatch: return "model shape mismatch";
        case ErrorCode::ModelUnsupportedArch:return "unsupported model architecture";
        case ErrorCode::RequestTooLong:   return "request too long";
        case ErrorCode::RequestCancelled: return "request cancelled";
        case ErrorCode::RequestTimeout:   return "request timed out";
        case ErrorCode::ServerShuttingDown: return "server shutting down";
    }
    return "unknown error";
}

} // namespace ccinfer
