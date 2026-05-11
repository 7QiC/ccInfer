#include "engine/model/loader.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

#include "engine/backend/cuda/cuda_utils.h"

namespace ccinfer {
namespace engine {

namespace {

Result<DType> parse_dtype(const std::string& dt) {
    if (dt == "BF16") return DType::kBFloat16;
    if (dt == "F16") return DType::kFloat16;
    if (dt == "F32") return DType::kFloat32;
    return std::unexpected(ErrorCode::ModelLoadFailed);
}

}  // namespace

WeightLoader::WeightLoader(std::string path) : path_(std::move(path)) {}

WeightLoader::~WeightLoader() { cleanup(); }

WeightLoader::WeightLoader(WeightLoader&& other) noexcept
    : path_(std::move(other.path_)),
      fd_(std::exchange(other.fd_, -1)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      header_len_(std::exchange(other.header_len_, 0)),
      tensors_(std::move(other.tensors_)) {}

WeightLoader& WeightLoader::operator=(WeightLoader&& other) noexcept {
    if (this != &other) {
        cleanup();
        path_ = std::move(other.path_);
        fd_ = std::exchange(other.fd_, -1);
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        header_len_ = std::exchange(other.header_len_, 0);
        tensors_ = std::move(other.tensors_);
    }
    return *this;
}

void WeightLoader::cleanup() noexcept {
    if (data_ != nullptr) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    size_ = 0;
    header_len_ = 0;
    tensors_.clear();
}

Result<WeightLoader> WeightLoader::create(const std::string& path) {
    WeightLoader loader(path);

    loader.fd_ = open(path.c_str(), O_RDONLY);
    if (loader.fd_ < 0) return std::unexpected(ErrorCode::ModelLoadFailed);

    struct stat st {};
    if (fstat(loader.fd_, &st) < 0) return std::unexpected(ErrorCode::ModelLoadFailed);
    if (st.st_size <= 0) return std::unexpected(ErrorCode::ModelLoadFailed);

    loader.size_ = static_cast<size_t>(st.st_size);

    void* mapped = mmap(nullptr, loader.size_, PROT_READ, MAP_PRIVATE, loader.fd_, 0);
    if (mapped == MAP_FAILED) return std::unexpected(ErrorCode::ModelLoadFailed);

    loader.data_ = static_cast<uint8_t*>(mapped);

    auto r = loader.parse();
    if (!r) return std::unexpected(r.error());

    return std::move(loader);
}

Result<void> WeightLoader::parse() {
    if (data_ == nullptr || size_ < kHeaderSize) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    uint64_t header_len = 0;
    std::memcpy(&header_len, data_, sizeof(uint64_t));

    if (header_len == 0 || header_len > size_ - kHeaderSize) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    header_len_ = header_len;

    const auto* header_begin = reinterpret_cast<const char*>(data_ + kHeaderSize);
    std::string header_str(header_begin, static_cast<size_t>(header_len_));

    auto j = nlohmann::json::parse(header_str, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    const uint64_t data_start = kHeaderSize + header_len_;

    for (auto& [name, info] : j.items()) {
        if (name == "__metadata__") continue;

        if (!info.is_object() || !info.contains("dtype") || !info.contains("shape") ||
            !info.contains("data_offsets")) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        if (!info["dtype"].is_string() || !info["shape"].is_array() ||
            !info["data_offsets"].is_array() || info["data_offsets"].size() != 2) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        TensorInfo t;

        auto dt = parse_dtype(info["dtype"].get<std::string>());
        if (!dt) return std::unexpected(dt.error());
        t.dtype_ = *dt;

        for (const auto& s : info["shape"]) {
            if (!s.is_number_integer()) return std::unexpected(ErrorCode::ModelLoadFailed);
            t.shape_.push_back(s.get<int64_t>());
        }

        if (!info["data_offsets"][0].is_number_unsigned() ||
            !info["data_offsets"][1].is_number_unsigned()) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }
        const uint64_t begin = info["data_offsets"][0].get<uint64_t>();
        const uint64_t end = info["data_offsets"][1].get<uint64_t>();

        if (end < begin) return std::unexpected(ErrorCode::ModelLoadFailed);

        t.offset_ = begin;
        t.size_bytes_ = end - begin;

        if (data_start > size_) return std::unexpected(ErrorCode::ModelLoadFailed);
        if (t.offset_ > size_ - data_start) return std::unexpected(ErrorCode::ModelLoadFailed);
        if (t.size_bytes_ > size_ - data_start - t.offset_) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        tensors_.emplace(name, std::move(t));
    }

    if (tensors_.empty()) return std::unexpected(ErrorCode::ModelLoadFailed);

    return {};
}

Result<TensorInfo> WeightLoader::info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return std::unexpected(ErrorCode::ModelLoadFailed);
    return it->second;
}

}  // namespace engine
}  // namespace ccinfer
