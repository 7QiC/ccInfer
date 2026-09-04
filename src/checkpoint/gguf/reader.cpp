#include "checkpoint/gguf/reader.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

#include "ccop/quant.h"

namespace ccinfer {

namespace {

constexpr std::array<uint8_t, 4> kMagic = {'G', 'G', 'U', 'F'};

Result<void> require_bytes(size_t size, size_t offset, size_t needed) {
    if (offset > size || needed > size - offset) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }
    return {};
}


template <typename T>
Result<T> read_raw(const uint8_t* data, size_t size, size_t& offset) {
    auto r = require_bytes(size, offset, sizeof(T));
    if (!r) return std::unexpected(r.error());
    T v;
    std::memcpy(&v, data + offset, sizeof(T));
    offset += sizeof(T);
    return v;
}

Result<uint8_t> read_u8(const uint8_t* data, size_t size, size_t& offset) {
    return read_raw<uint8_t>(data, size, offset);
}

Result<int8_t> read_i8(const uint8_t* data, size_t size, size_t& offset) {
    return read_raw<int8_t>(data, size, offset);
}

Result<uint16_t> read_u16(const uint8_t* data, size_t size, size_t& offset) {
    return read_raw<uint16_t>(data, size, offset);
}

Result<int16_t> read_i16(const uint8_t* data, size_t size, size_t& offset) {
    return read_raw<int16_t>(data, size, offset);
}

Result<uint32_t> read_u32(const uint8_t* data, size_t size, size_t& offset) {
    return read_raw<uint32_t>(data, size, offset);
}

Result<int32_t> read_i32(const uint8_t* data, size_t size, size_t& offset) {
    return read_raw<int32_t>(data, size, offset);
}

Result<uint64_t> read_u64(const uint8_t* data, size_t size, size_t& offset) {
    return read_raw<uint64_t>(data, size, offset);
}

Result<int64_t> read_i64(const uint8_t* data, size_t size, size_t& offset) {
    return read_raw<int64_t>(data, size, offset);
}

Result<float> read_f32(const uint8_t* data, size_t size, size_t& offset) {
    return read_raw<float>(data, size, offset);
}

Result<double> read_f64(const uint8_t* data, size_t size, size_t& offset) {
    return read_raw<double>(data, size, offset);
}

Result<void> read_bool(const uint8_t* data, size_t size, size_t& offset, bool& out) {
    auto v_r = read_u8(data, size, offset);
    if (!v_r) return std::unexpected(v_r.error());
    out = *v_r != 0;
    return {};
}

Result<std::string> read_string(const uint8_t* data, size_t size, size_t& offset) {
    auto len_r = read_u64(data, size, offset);
    if (!len_r) return std::unexpected(len_r.error());
    if (*len_r > std::numeric_limits<size_t>::max()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }
    const auto len = static_cast<size_t>(*len_r);
    auto r = require_bytes(size, offset, len);
    if (!r) return std::unexpected(r.error());
    std::string s(reinterpret_cast<const char*>(data + offset), len);
    offset += len;
    return s;
}


bool supported_tensor_type(GGUFTensorType type) {
    switch (type) {
        case GGUFTensorType::kF32:
        case GGUFTensorType::kF16:
        case GGUFTensorType::kQ8_0:
        case GGUFTensorType::kBF16:
            return true;
        default:
            return false;
    }
}

Result<GGUFMetadataValue> parse_value(const uint8_t* data, size_t size, size_t& offset,
                                      GGUFValueType type) {
    GGUFMetadataValue result;
    result.type = type;

    switch (type) {
        case GGUFValueType::kU8: {
            auto v_r = read_u8(data, size, offset);
            if (!v_r) return std::unexpected(v_r.error());
            result.value = *v_r;
            return result;
        }
        case GGUFValueType::kI8: {
            auto v_r = read_i8(data, size, offset);
            if (!v_r) return std::unexpected(v_r.error());
            result.value = *v_r;
            return result;
        }
        case GGUFValueType::kU16: {
            auto v_r = read_u16(data, size, offset);
            if (!v_r) return std::unexpected(v_r.error());
            result.value = *v_r;
            return result;
        }
        case GGUFValueType::kI16: {
            auto v_r = read_i16(data, size, offset);
            if (!v_r) return std::unexpected(v_r.error());
            result.value = *v_r;
            return result;
        }
        case GGUFValueType::kU32: {
            auto v_r = read_u32(data, size, offset);
            if (!v_r) return std::unexpected(v_r.error());
            result.value = *v_r;
            return result;
        }
        case GGUFValueType::kI32: {
            auto v_r = read_i32(data, size, offset);
            if (!v_r) return std::unexpected(v_r.error());
            result.value = *v_r;
            return result;
        }
        case GGUFValueType::kU64: {
            auto v_r = read_u64(data, size, offset);
            if (!v_r) return std::unexpected(v_r.error());
            result.value = *v_r;
            return result;
        }
        case GGUFValueType::kI64: {
            auto v_r = read_i64(data, size, offset);
            if (!v_r) return std::unexpected(v_r.error());
            result.value = *v_r;
            return result;
        }
        case GGUFValueType::kF32: {
            auto v_r = read_f32(data, size, offset);
            if (!v_r) return std::unexpected(v_r.error());
            result.value = *v_r;
            return result;
        }
        case GGUFValueType::kF64: {
            auto v_r = read_f64(data, size, offset);
            if (!v_r) return std::unexpected(v_r.error());
            result.value = *v_r;
            return result;
        }
        case GGUFValueType::kBool: {
            bool v = false;
            auto r = read_bool(data, size, offset, v);
            if (!r) return std::unexpected(r.error());
            result.value = v;
            return result;
        }
        case GGUFValueType::kString: {
            auto s_r = read_string(data, size, offset);
            if (!s_r) return std::unexpected(s_r.error());
            result.value = std::move(*s_r);
            return result;
        }
        case GGUFValueType::kArray: {
            auto elem_type_r = read_u32(data, size, offset);
            if (!elem_type_r) return std::unexpected(elem_type_r.error());
            auto count_r = read_u64(data, size, offset);
            if (!count_r) return std::unexpected(count_r.error());

            std::vector<GGUFMetadataValue> values;
            values.reserve(static_cast<size_t>(*count_r));
            for (uint64_t i = 0; i < *count_r; ++i) {
                auto v_r =
                    parse_value(data, size, offset, static_cast<GGUFValueType>(*elem_type_r));
                if (!v_r) return std::unexpected(v_r.error());
                values.push_back(std::move(*v_r));
            }
            result.value = std::move(values);
            return result;
        }
    }

    return std::unexpected(ErrorCode::ModelLoadFailed);
}

int64_t tensor_numel(const GGUFTensorInfo& info) {
    if (info.dims.empty()) return 0;
    int64_t n = 1;
    for (int64_t dim : info.dims) {
        if (dim <= 0) return 0;
        if (n > std::numeric_limits<int64_t>::max() / dim) return 0;
        n *= dim;
    }
    return n;
}

}  // namespace

const std::string& GGUFMetadataValue::as_string() const { return std::get<std::string>(value); }

std::optional<uint64_t> GGUFMetadataValue::as_u64() const {
    if (const auto* v = std::get_if<uint64_t>(&value)) return *v;
    if (const auto* v = std::get_if<uint32_t>(&value)) return *v;
    if (const auto* v = std::get_if<uint16_t>(&value)) return *v;
    if (const auto* v = std::get_if<uint8_t>(&value)) return *v;
    return std::nullopt;
}

std::optional<int64_t> GGUFMetadataValue::as_i64() const {
    if (const auto* v = std::get_if<int64_t>(&value)) return std::optional<int64_t>{*v};
    if (const auto* v = std::get_if<int32_t>(&value)) return std::optional<int64_t>{*v};
    if (const auto* v = std::get_if<int16_t>(&value)) return std::optional<int64_t>{*v};
    if (const auto* v = std::get_if<int8_t>(&value)) return std::optional<int64_t>{*v};
    return as_u64().transform([](uint64_t x) { return static_cast<int64_t>(x); });
}

std::optional<double> GGUFMetadataValue::as_f64() const {
    if (const auto* v = std::get_if<double>(&value)) return *v;
    if (const auto* v = std::get_if<float>(&value)) return *v;
    return std::nullopt;
}

std::optional<float> GGUFMetadataValue::as_f32() const {
    if (const auto* v = std::get_if<float>(&value)) return *v;
    if (const auto* v = std::get_if<double>(&value)) return static_cast<float>(*v);
    return std::nullopt;
}

const std::vector<GGUFMetadataValue>* GGUFMetadataValue::as_array() const {
    return std::get_if<std::vector<GGUFMetadataValue>>(&value);
}

GGUFReader::~GGUFReader() {
    if (data_ != nullptr) munmap(const_cast<uint8_t*>(data_), size_);
    data_ = nullptr;
}

Result<std::unique_ptr<GGUFReader>> GGUFReader::create(const std::string& path) {
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

    auto reader = std::unique_ptr<GGUFReader>(new GGUFReader());
    reader->data_ = static_cast<const uint8_t*>(mapped);
    reader->size_ = static_cast<size_t>(st.st_size);

    auto r = reader->parse(reader->data_, reader->size_);
    if (!r) {
        reader.reset();
        return std::unexpected(r.error());
    }
    return reader;
}

Result<void> GGUFReader::parse(const uint8_t* data, size_t size) {
    if (size < 4 + 4 + 8 + 8) return std::unexpected(ErrorCode::ModelLoadFailed);
    if (std::memcmp(data, kMagic.data(), 4) != 0) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    size_t offset = 0;
    offset += 4;
    uint32_t version;
    std::memcpy(&version, data + offset, 4);
    offset += 4;
    if (version != 3) return std::unexpected(ErrorCode::ModelLoadFailed);
    version_ = version;

    auto n_tensors_r = read_u64(data, size, offset);
    if (!n_tensors_r) return std::unexpected(n_tensors_r.error());
    const uint64_t n_tensors = *n_tensors_r;
    auto n_metadata_r = read_u64(data, size, offset);
    if (!n_metadata_r) return std::unexpected(n_metadata_r.error());
    const uint64_t n_metadata = *n_metadata_r;

    metadata_.reserve(static_cast<size_t>(n_metadata));
    for (uint64_t i = 0; i < n_metadata; ++i) {
        auto key_r = read_string(data, size, offset);
        if (!key_r) return std::unexpected(key_r.error());
        auto type_r = read_u32(data, size, offset);
        if (!type_r) return std::unexpected(type_r.error());
        const auto type = static_cast<GGUFValueType>(*type_r);
        if (type > GGUFValueType::kF64) return std::unexpected(ErrorCode::ModelLoadFailed);
        auto value_r = parse_value(data, size, offset, type);
        if (!value_r) return std::unexpected(value_r.error());
        metadata_.emplace_back(std::move(*key_r), std::move(*value_r));
    }

    tensor_infos_.reserve(static_cast<size_t>(n_tensors));
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GGUFTensorInfo info;
        auto name_r = read_string(data, size, offset);
        if (!name_r) return std::unexpected(name_r.error());
        info.name = std::move(*name_r);

        auto n_dims_r = read_u32(data, size, offset);
        if (!n_dims_r) return std::unexpected(n_dims_r.error());
        if (*n_dims_r == 0 || *n_dims_r > 8) return std::unexpected(ErrorCode::ModelLoadFailed);

        for (uint32_t d = 0; d < *n_dims_r; ++d) {
            auto dim_r = read_u64(data, size, offset);
            if (!dim_r) return std::unexpected(dim_r.error());
            info.dims.push_back(static_cast<int64_t>(*dim_r));
        }

        auto type_r = read_u32(data, size, offset);
        if (!type_r) return std::unexpected(type_r.error());
        if (*type_r > static_cast<uint32_t>(GGUFTensorType::kBF16)) {
            return std::unexpected(ErrorCode::ModelUnsupportedDType);
        }
        info.type = static_cast<GGUFTensorType>(*type_r);
        if (!supported_tensor_type(info.type)) {
            return std::unexpected(ErrorCode::ModelUnsupportedDType);
        }

        auto tensor_offset_r = read_u64(data, size, offset);
        if (!tensor_offset_r) return std::unexpected(tensor_offset_r.error());
        info.offset = *tensor_offset_r;

        tensor_infos_.push_back(std::move(info));
    }

    data_start_ = offset;
    if (data_start_ > size) return std::unexpected(ErrorCode::ModelLoadFailed);
    return {};
}

std::optional<GGUFMetadataValue> GGUFReader::metadata(std::string_view key) const {
    for (const auto& [k, v] : metadata_) {
        if (k == key) return v;
    }
    return std::nullopt;
}

std::optional<GGUFTensorInfo> GGUFReader::find_tensor(std::string_view name) const {
    for (const auto& info : tensor_infos_) {
        if (info.name == name) return info;
    }
    return std::nullopt;
}

Result<uint64_t> GGUFReader::tensor_bytes(const GGUFTensorInfo& info) const {
    switch (info.type) {
        case GGUFTensorType::kF32: {
            const int64_t n = tensor_numel(info);
            if (n <= 0) return std::unexpected(ErrorCode::ModelShapeMismatch);
            return static_cast<uint64_t>(n) * sizeof(float);
        }
        case GGUFTensorType::kF16:
        case GGUFTensorType::kBF16: {
            const int64_t n = tensor_numel(info);
            if (n <= 0) return std::unexpected(ErrorCode::ModelShapeMismatch);
            return static_cast<uint64_t>(n) * 2;
        }
        case GGUFTensorType::kQ8_0: {
            // GGUF raw dims are in reverse logical order: ne[0] is the
            // block-aligned innermost dimension. q8_0_storage_bytes expects a
            // canonical logical row-major shape (LAST dim block-aligned), so
            // reverse the raw dims before computing physical bytes.
            auto logical = info.dims;
            std::reverse(logical.begin(), logical.end());
            auto bytes = ccop::q8_0_storage_bytes(logical);
            if (!bytes.has_value()) return std::unexpected(ErrorCode::ModelShapeMismatch);
            return static_cast<uint64_t>(*bytes);
        }
        default:
            return std::unexpected(ErrorCode::Unsupported);
    }
}

Result<std::span<const uint8_t>> GGUFReader::raw_tensor_data(const GGUFTensorInfo& info) const {
    auto bytes_r = tensor_bytes(info);
    if (!bytes_r) return std::unexpected(bytes_r.error());

    const uint64_t data_offset = data_start_ + info.offset;
    if (data_offset > size_ || *bytes_r > size_ - data_offset) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }
    return std::span<const uint8_t>(data_ + data_offset, static_cast<size_t>(*bytes_r));
}

bool GGUFReader::has_nextn_tensors() const noexcept {
    for (const auto& info : tensor_infos_) {
        if (info.name.find(".nextn.") != std::string::npos ||
            info.name.find("nextn_") != std::string::npos) {
            return true;
        }
    }
    return false;
}

}  // namespace ccinfer
