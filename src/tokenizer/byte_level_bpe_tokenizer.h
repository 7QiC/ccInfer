#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tokenizer/tokenizer.h"

namespace ccinfer {

class ByteLevelBpeTokenizer final : public Tokenizer {
public:
    ByteLevelBpeTokenizer() = default;

    Result<void> load(const std::string& path) override;

    Result<std::vector<int32_t>> encode(std::string_view text,
                                        bool add_special_tokens = false) const override;

    Result<std::string> decode(const std::vector<int32_t>& token_ids,
                               bool skip_special_tokens = true) const override;

    int32_t bos_token_id() const noexcept override { return bos_token_id_; }
    int32_t eos_token_id() const noexcept override { return eos_token_id_; }
    int32_t pad_token_id() const noexcept override { return pad_token_id_; }
    int32_t unk_token_id() const noexcept override { return unk_token_id_; }

    int32_t vocab_size() const noexcept override { return max_token_id_ + 1; }

private:
    struct Pair {
        std::string a;
        std::string b;

        bool operator==(const Pair& other) const { return a == other.a && b == other.b; }
    };

    struct PairHash {
        size_t operator()(const Pair& p) const {
            const size_t h1 = std::hash<std::string>{}(p.a);
            const size_t h2 = std::hash<std::string>{}(p.b);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    void build_byte_maps();

    Result<std::vector<std::string>> byte_encode_segment(std::string_view text) const;
    Result<std::vector<std::string>> apply_bpe(std::vector<std::string> symbols) const;
    Result<std::vector<int32_t>> encode_normal_segment(std::string_view text) const;

    bool try_match_special(std::string_view text, size_t pos, std::string* matched_token,
                           int32_t* matched_id) const;

    static std::string utf8_from_codepoint(uint32_t cp);

private:
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<int32_t, std::string> id_to_token_;

    std::unordered_map<Pair, int32_t, PairHash> merge_rank_;

    std::unordered_map<uint8_t, std::string> byte_to_str_;
    std::unordered_map<std::string, uint8_t> str_to_byte_;

    std::unordered_map<std::string, int32_t> special_token_to_id_;
    std::unordered_map<int32_t, std::string> id_to_special_token_;

    int32_t max_token_id_ = -1;
    int32_t bos_token_id_ = -1;
    int32_t eos_token_id_ = -1;
    int32_t pad_token_id_ = -1;
    int32_t unk_token_id_ = -1;
};

}  // namespace ccinfer
