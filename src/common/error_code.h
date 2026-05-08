#pragma once

#include <boost/system/error_code.hpp>

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
    }

    return "unknown error";
}

class ErrorCategory : public boost::system::error_category {
public:
    const char* name() const noexcept override { return "ccinfer"; }
    std::string message(int ev) const override {
        return std::string(error_message(static_cast<ErrorCode>(ev)));
    }
};

inline const boost::system::error_category& engine_error_category() {
    static ErrorCategory instance;
    return instance;
}

inline boost::system::error_code make_error_code(ErrorCode c) {
    return {static_cast<int>(c), engine_error_category()};
}

}  // namespace ccinfer

namespace boost {
namespace system {
template <>
struct is_error_code_enum<ccinfer::ErrorCode> : std::true_type {};
}  // namespace system
}  // namespace boost
