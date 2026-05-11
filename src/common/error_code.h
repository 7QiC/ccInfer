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

    KVBlockExhausted,
    KVBlockDoubleFree,
    KVInvalidBlockTable,

    RequestTooLong,
    RequestCancelled,
    RequestTimeout,
    ServerShuttingDown,

    MaxSequencesReached,
    BatchTranslationFailed,

    ChannelClosed,
    ChannelCancelled,
    ChannelError,

    NetworkBindFailed,
    Unsupported,
    InternalError,
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

        case ErrorCode::KVBlockExhausted:
            return "KV cache block exhausted";
        case ErrorCode::KVBlockDoubleFree:
            return "KV cache block double-free";
        case ErrorCode::KVInvalidBlockTable:
            return "invalid KV cache block table";

        case ErrorCode::RequestTooLong:
            return "request too long";
        case ErrorCode::RequestCancelled:
            return "request cancelled";
        case ErrorCode::RequestTimeout:
            return "request timed out";
        case ErrorCode::ServerShuttingDown:
            return "server shutting down";

        case ErrorCode::MaxSequencesReached:
            return "maximum concurrent sequences reached";
        case ErrorCode::BatchTranslationFailed:
            return "WorkItem to PhysicalBatch translation failed";

        case ErrorCode::ChannelClosed:
            return "asio channel closed";
        case ErrorCode::ChannelCancelled:
            return "asio channel operation cancelled";
        case ErrorCode::ChannelError:
            return "asio channel error";

        case ErrorCode::NetworkBindFailed:
            return "network bind failed";

        case ErrorCode::Unsupported:
            return "unsupported operation";
        case ErrorCode::InternalError:
            return "internal error";
    }

    return "unknown error";
}

}  // namespace ccinfer
