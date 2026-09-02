#include "checkpoint/huggingface/reader.h"

#include <cstring>
#include <fcntl.h>
#include <nlohmann/json.hpp>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace ccinfer {

namespace {

Result<ccop::DType> parse_dtype(const std::string& dt) {
    if (dt == "BF16") return ccop::DType::kBFloat16;
    if (dt == "F16") return ccop::DType::kFloat16;
    if (dt == "F32") return ccop::DType::kFloat32;
    return std::unexpected(ErrorCode::ModelLoadFailed);
}

}  // namespace

Reader::~Reader() {
    if (data_ != nullptr) munmap(const_cast<uint8_t*>(data_), size_);
    data_ = nullptr;
}

Result<std::unique_ptr<Reader>> Reader::create(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return std::unexpected(ErrorCode::ModelLoadFailed);

    struct stat st {};
    if (fstat(fd, &st) < 0 || st.st_size <= 0) {
        close(fd);
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    void* mapped = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) return std::unexpected(ErrorCode::ModelLoadFailed);

    auto reader = std::unique_ptr<Reader>(new Reader());
    reader->data_ = static_cast<const uint8_t*>(mapped);
    reader->size_ = static_cast<size_t>(st.st_size);

    auto r = reader->parse();
    if (!r) {
        reader.reset();
        return std::unexpected(r.error());
    }
    return reader;
}

Result<void> Reader::parse() {
    if (data_ == nullptr || size_ < kHeaderSize) return std::unexpected(ErrorCode::ModelLoadFailed);

    uint64_t header_len = 0;
    std::memcpy(&header_len, data_, sizeof(uint64_t));
    if (header_len == 0 || header_len > size_ - kHeaderSize) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    const auto* header_begin = reinterpret_cast<const char*>(data_ + kHeaderSize);
    std::string header_str(header_begin, static_cast<size_t>(header_len));
    auto j = nlohmann::json::parse(header_str, nullptr, false);
    if (j.is_discarded() || !j.is_object()) return std::unexpected(ErrorCode::ModelLoadFailed);

    const uint64_t data_start = kHeaderSize + header_len;
    if (data_start > size_) return std::unexpected(ErrorCode::ModelLoadFailed);
    data_start_ = data_start;

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

        SafetensorsTensorInfo t;
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

        if (t.offset_ > size_ - data_start) return std::unexpected(ErrorCode::ModelLoadFailed);
        if (t.size_bytes_ > size_ - data_start - t.offset_) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        ordered_tensors_.push_back(t);
        tensors_.emplace(name, std::move(t));
    }

    if (tensors_.empty()) return std::unexpected(ErrorCode::ModelLoadFailed);
    return {};
}

bool Reader::has(std::string_view name) const {
    return tensors_.find(std::string(name)) != tensors_.end();
}

std::optional<SafetensorsTensorInfo> Reader::find_tensor(std::string_view name) const {
    auto it = tensors_.find(std::string(name));
    if (it == tensors_.end()) return std::nullopt;
    return it->second;
}

const std::vector<SafetensorsTensorInfo>& Reader::tensors() const noexcept {
    return ordered_tensors_;
}

Result<std::span<const uint8_t>> Reader::raw_tensor_data(
    const SafetensorsTensorInfo& info) const {
    if (info.offset_ > size_ - data_start_ ||
        info.size_bytes_ > size_ - data_start_ - info.offset_) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }
    return std::span<const uint8_t>(data_ + data_start_ + info.offset_,
                                    static_cast<size_t>(info.size_bytes_));
}

}  // namespace ccinfer
