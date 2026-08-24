#include <cstddef>
#include <cublas_v2.h>
#include <memory>
#include <utility>

#include <cuda_runtime.h>

#include "backend/backend.h"
#include "backend/cuda/cuda_utils.h"
#include "common/error_code.h"

namespace ccinfer {

namespace {

// CUDA-backed Buffer implementation. Framework code only sees Buffer;
// this class is an internal detail of Backend.
class CudaBuffer final : public Buffer {
public:
    static Result<std::shared_ptr<CudaBuffer>> create(std::size_t bytes, ccop::Device device) {
        if (bytes == 0) return std::unexpected(ErrorCode::InvalidArgument);
        void* ptr = nullptr;
        auto r = cuda_check(cudaMalloc(&ptr, bytes));
        if (!r) return std::unexpected(r.error());
        return std::shared_ptr<CudaBuffer>(new CudaBuffer(bytes, ptr, device));
    }

    ~CudaBuffer() override {
        if (ptr_) cudaFree(ptr_);
    }

    void* data() noexcept override { return ptr_; }
    const void* data() const noexcept override { return ptr_; }
    std::size_t bytes() const noexcept override { return bytes_; }
    ccop::Device device() const noexcept override { return device_; }

private:
    CudaBuffer(std::size_t bytes, void* ptr, ccop::Device device)
        : bytes_(bytes), ptr_(ptr), device_(device) {}

    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
    ccop::Device device_{};
};

}  // namespace

struct Backend::Impl {
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
    int device_id_ = 0;
};

Result<std::unique_ptr<Backend>> Backend::create(int device_id) {
    auto impl = std::make_unique<Impl>();

    if (auto r = cuda_check(cudaSetDevice(device_id)); !r) return std::unexpected(r.error());
    impl->device_id_ = device_id;
    if (auto r = cublas_check(cublasCreate(&impl->cublas_handle_)); !r) {
        return std::unexpected(r.error());
    }
    if (auto r = cuda_check(cudaStreamCreate(&impl->stream_)); !r) {
        cublasDestroy(impl->cublas_handle_);
        return std::unexpected(r.error());
    }
    if (auto r = cublas_check(cublasSetStream(impl->cublas_handle_, impl->stream_)); !r) {
        cudaStreamDestroy(impl->stream_);
        cublasDestroy(impl->cublas_handle_);
        return std::unexpected(r.error());
    }

    return std::unique_ptr<Backend>(new Backend(std::move(impl)));
}

Backend::Backend(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Backend::~Backend() = default;
Backend::Backend(Backend&&) noexcept = default;
Backend& Backend::operator=(Backend&&) noexcept = default;

Result<std::shared_ptr<Buffer>> Backend::allocate_buffer(std::size_t bytes) {
    if (bytes == 0) return std::unexpected(ErrorCode::InvalidArgument);
    auto buf = CudaBuffer::create(bytes, ccop::Device{ccop::DeviceType::kCUDA, impl_->device_id_});
    if (!buf) return std::unexpected(buf.error());
    return std::move(*buf);
}

Result<void> Backend::memcpy_h2d(void* dst, const void* src, std::size_t count) {
    if (count == 0) return {};
    if (dst == nullptr || src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
    return cuda_check(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, impl_->stream_));
}

Result<void> Backend::memcpy_d2h(void* dst, const void* src, std::size_t count) {
    if (count == 0) return {};
    if (dst == nullptr || src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
    if (auto r =
            cuda_check(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, impl_->stream_));
        !r)
        return r;
    return cuda_check(cudaStreamSynchronize(impl_->stream_));
}

Result<void> Backend::memcpy_d2d(void* dst, const void* src, std::size_t count) {
    if (count == 0) return {};
    if (dst == nullptr || src == nullptr) return std::unexpected(ErrorCode::InvalidArgument);
    return cuda_check(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, impl_->stream_));
}

void* Backend::stream() const noexcept { return impl_->stream_; }

Result<void> Backend::synchronize() { return cuda_check(cudaStreamSynchronize(impl_->stream_)); }

ccop::ExecutionContext Backend::context() const noexcept {
    return {impl_->stream_, impl_->cublas_handle_};
}

}  // namespace ccinfer
