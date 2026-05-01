#include "engine/tokenizer/byte_level_bpe_tokenizer.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <utility>

#include "common/error_code.h"

namespace ccinfer {
namespace engine {

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return {};
    }

    f.seekg(0, std::ios::end);
    const auto end = f.tellg();
    if (end <= 0) {
        return {};
    }

    std::string content;
    content.resize(static_cast<size_t>(end));

    f.seekg(0, std::ios::beg);
    f.read(content.data(), static_cast<std::streamsize>(content.size()));

    if (!f) {
        return {};
    }

    return content;
}

bool starts_with_at(std::string_view text, size_t pos, std::string_view pattern) {
    if (pos + pattern.size() > text.size()) {
        return false;
    }

    return text.substr(pos, pattern.size()) == pattern;
}

}  // namespace

std::string ByteLevelBpeTokenizer::utf8_from_codepoint(uint32_t cp) {
    std::string s;

    if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }

    return s;
}

void ByteLevelBpeTokenizer::build_byte_maps() {
    std::vector<uint8_t> bytes;
    std::vector<uint32_t> codepoints;

    auto add_range = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b) {
            bytes.push_back(static_cast<uint8_t>(b));
            codepoints.push_back(static_cast<uint32_t>(b));
        }
    };

    // GPT-2 byte-to-unicode mapping.
    add_range(33, 126);
    add_range(161, 172);
    add_range(174, 255);

    uint32_t n = 0;
    for (int b = 0; b < 256; ++b) {
        const auto ub = static_cast<uint8_t>(b);
        if (std::find(bytes.begin(), bytes.end(), ub) == bytes.end()) {
            bytes.push_back(ub);
            codepoints.push_back(256 + n);
            ++n;
        }
    }

    byte_to_str_.clear();
    str_to_byte_.clear();

    for (size_t i = 0; i < bytes.size(); ++i) {
        std::string s = utf8_from_codepoint(codepoints[i]);
        byte_to_str_[bytes[i]] = s;
        str_to_byte_[s] = bytes[i];
    }
}

Result<void> ByteLevelBpeTokenizer::load(const std::string& path) {
    const std::string data = read_file(path);
    if (data.empty()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    auto j = nlohmann::json::parse(data, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    if (!j.contains("model") || !j["model"].is_object()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    const auto& model = j["model"];

    if (model.contains("type") && model["type"].is_string()) {
        const std::string type = model["type"].get<std::string>();
        if (type != "BPE") {
            return std::unexpected(ErrorCode::ModelUnsupportedArch);
        }
    }

    if (!model.contains("vocab") || !model["vocab"].is_object()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    vocab_.clear();
    id_to_token_.clear();
    merge_rank_.clear();
    special_token_to_id_.clear();
    id_to_special_token_.clear();

    for (const auto& [token, id_val] : model["vocab"].items()) {
        if (!id_val.is_number_integer()) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        const int32_t id = id_val.get<int32_t>();
        vocab_[token] = id;
        id_to_token_[id] = token;
    }

    if (model.contains("merges") && model["merges"].is_array()) {
        int32_t rank = 0;

        for (const auto& merge_item : model["merges"]) {
            std::string a;
            std::string b;

            if (merge_item.is_string()) {
                const std::string merge = merge_item.get<std::string>();
                const auto pos = merge.find(' ');
                if (pos == std::string::npos) {
                    continue;
                }

                a = merge.substr(0, pos);
                b = merge.substr(pos + 1);
            } else if (merge_item.is_array() && merge_item.size() == 2 &&
                       merge_item[0].is_string() && merge_item[1].is_string()) {
                a = merge_item[0].get<std::string>();
                b = merge_item[1].get<std::string>();
            } else {
                return std::unexpected(ErrorCode::ModelLoadFailed);
            }

            merge_rank_[Pair{std::move(a), std::move(b)}] = rank++;
        }
    }

    // Parse added/special tokens from tokenizer.json.
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (const auto& tok : j["added_tokens"]) {
            if (!tok.is_object()) {
                continue;
            }

            if (!tok.contains("content") || !tok.contains("id")) {
                continue;
            }

            if (!tok["content"].is_string() || !tok["id"].is_number_integer()) {
                continue;
            }

            const std::string content = tok["content"].get<std::string>();
            const int32_t id = tok["id"].get<int32_t>();

            special_token_to_id_[content] = id;
            id_to_special_token_[id] = content;
            vocab_[content] = id;
            id_to_token_[id] = content;

            if (content == "<s>" || content == "<|begin_of_text|>") {
                bos_token_id_ = id;
            } else if (content == "</s>" || content == "<|end_of_text|>" ||
                       content == "<|endoftext|>") {
                eos_token_id_ = id;
            } else if (content == "<pad>" || content == "[PAD]") {
                pad_token_id_ = id;
            } else if (content == "<unk>" || content == "[UNK]") {
                unk_token_id_ = id;
            }
        }
    }

    // Some tokenizer.json files store special token strings only through vocab.
    auto set_if_exists = [&](const std::string& token, int32_t* dst) {
        auto it = vocab_.find(token);
        if (it != vocab_.end() && *dst < 0) {
            *dst = it->second;
            special_token_to_id_[token] = it->second;
            id_to_special_token_[it->second] = token;
        }
    };

    set_if_exists("<s>", &bos_token_id_);
    set_if_exists("<|begin_of_text|>", &bos_token_id_);

    set_if_exists("</s>", &eos_token_id_);
    set_if_exists("<|end_of_text|>", &eos_token_id_);
    set_if_exists("<|endoftext|>", &eos_token_id_);

    set_if_exists("<pad>", &pad_token_id_);
    set_if_exists("[PAD]", &pad_token_id_);

    set_if_exists("<unk>", &unk_token_id_);
    set_if_exists("[UNK]", &unk_token_id_);

    build_byte_maps();

    if (vocab_.empty()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    return {};
}

Result<std::vector<std::string>> ByteLevelBpeTokenizer::byte_encode_segment(
    std::string_view text) const {
    std::vector<std::string> symbols;
    symbols.reserve(text.size());

    const auto* bytes = reinterpret_cast<const uint8_t*>(text.data());

    for (size_t i = 0; i < text.size(); ++i) {
        auto it = byte_to_str_.find(bytes[i]);
        if (it == byte_to_str_.end()) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        symbols.push_back(it->second);
    }

    return symbols;
}

Result<std::vector<std::string>> ByteLevelBpeTokenizer::apply_bpe(
    std::vector<std::string> symbols) const {
    if (symbols.empty()) {
        return symbols;
    }

    while (symbols.size() >= 2) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best_pos = symbols.size();

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            Pair p{symbols[i], symbols[i + 1]};
            auto it = merge_rank_.find(p);

            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }

        if (best_pos == symbols.size()) {
            break;
        }

        std::vector<std::string> next;
        next.reserve(symbols.size() - 1);

        size_t i = 0;
        while (i < symbols.size()) {
            if (i + 1 < symbols.size()) {
                Pair p{symbols[i], symbols[i + 1]};
                auto it = merge_rank_.find(p);

                if (it != merge_rank_.end() && it->second == best_rank) {
                    next.push_back(symbols[i] + symbols[i + 1]);
                    i += 2;
                    continue;
                }
            }

            next.push_back(std::move(symbols[i]));
            ++i;
        }

        symbols = std::move(next);
    }

    return symbols;
}

Result<std::vector<int32_t>> ByteLevelBpeTokenizer::encode_normal_segment(
    std::string_view text) const {
    auto symbols_result = byte_encode_segment(text);
    if (!symbols_result) {
        return std::unexpected(symbols_result.error());
    }

    auto bpe_result = apply_bpe(std::move(*symbols_result));
    if (!bpe_result) {
        return std::unexpected(bpe_result.error());
    }

    const std::vector<std::string>& symbols = *bpe_result;

    std::vector<int32_t> ids;
    ids.reserve(symbols.size());

    for (const auto& s : symbols) {
        auto it = vocab_.find(s);
        if (it == vocab_.end()) {
            if (unk_token_id_ >= 0) {
                ids.push_back(unk_token_id_);
                continue;
            }

            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        ids.push_back(it->second);
    }

    return ids;
}

bool ByteLevelBpeTokenizer::try_match_special(std::string_view text, size_t pos,
                                              std::string* matched_token,
                                              int32_t* matched_id) const {
    size_t best_len = 0;
    int32_t best_id = -1;
    std::string best_token;

    for (const auto& [tok, id] : special_token_to_id_) {
        if (tok.empty()) {
            continue;
        }

        if (tok.size() <= best_len) {
            continue;
        }

        if (starts_with_at(text, pos, tok)) {
            best_len = tok.size();
            best_id = id;
            best_token = tok;
        }
    }

    if (best_len == 0) {
        return false;
    }

    *matched_token = std::move(best_token);
    *matched_id = best_id;
    return true;
}

Result<std::vector<int32_t>> ByteLevelBpeTokenizer::encode(std::string_view text,
                                                           bool add_special_tokens) const {
    if (vocab_.empty()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    std::vector<int32_t> ids;

    if (add_special_tokens && bos_token_id_ >= 0) {
        ids.push_back(bos_token_id_);
    }

    size_t pos = 0;
    size_t segment_start = 0;

    auto flush_segment = [&](size_t begin, size_t end) -> Result<void> {
        if (end <= begin) {
            return {};
        }

        auto seg_ids = encode_normal_segment(text.substr(begin, end - begin));
        if (!seg_ids) {
            return std::unexpected(seg_ids.error());
        }

        ids.insert(ids.end(), seg_ids->begin(), seg_ids->end());
        return {};
    };

    while (pos < text.size()) {
        std::string matched_token;
        int32_t matched_id = -1;

        if (try_match_special(text, pos, &matched_token, &matched_id)) {
            auto r = flush_segment(segment_start, pos);
            if (!r) {
                return std::unexpected(r.error());
            }

            ids.push_back(matched_id);
            pos += matched_token.size();
            segment_start = pos;
            continue;
        }

        ++pos;
    }

    auto r = flush_segment(segment_start, text.size());
    if (!r) {
        return std::unexpected(r.error());
    }

    if (add_special_tokens && eos_token_id_ >= 0) {
        ids.push_back(eos_token_id_);
    }

    return ids;
}

Result<std::string> ByteLevelBpeTokenizer::decode(const std::vector<int32_t>& token_ids,
                                                  bool skip_special_tokens) const {
    if (id_to_token_.empty()) {
        return std::unexpected(ErrorCode::ModelLoadFailed);
    }

    std::string symbol_stream;

    for (int32_t id : token_ids) {
        auto special_it = id_to_special_token_.find(id);
        if (special_it != id_to_special_token_.end()) {
            if (!skip_special_tokens) {
                symbol_stream += special_it->second;
            }
            continue;
        }

        auto it = id_to_token_.find(id);
        if (it == id_to_token_.end()) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        symbol_stream += it->second;
    }

    std::string result;
    size_t i = 0;

    while (i < symbol_stream.size()) {
        const auto c = static_cast<uint8_t>(symbol_stream[i]);

        size_t len = 1;
        if ((c & 0x80) == 0) {
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        } else {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        if (i + len > symbol_stream.size()) {
            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        std::string ch = symbol_stream.substr(i, len);

        auto it = str_to_byte_.find(ch);
        if (it == str_to_byte_.end()) {
            // This can happen if skip_special_tokens=false and a special token
            // was appended literally. In that case, preserve literal bytes.
            if (!skip_special_tokens) {
                result += ch;
                i += len;
                continue;
            }

            return std::unexpected(ErrorCode::ModelLoadFailed);
        }

        result.push_back(static_cast<char>(it->second));
        i += len;
    }

    return result;
}

Result<std::unique_ptr<Tokenizer>> create_tokenizer(const std::string& model_dir) {
    namespace fs = std::filesystem;

    const fs::path dir(model_dir);
    const fs::path tokenizer_json = dir / "tokenizer.json";

    if (fs::exists(tokenizer_json)) {
        auto tokenizer = std::make_unique<ByteLevelBpeTokenizer>();

        auto r = tokenizer->load(tokenizer_json.string());
        if (!r) {
            return std::unexpected(r.error());
        }

        return tokenizer;
    }

    return std::unexpected(ErrorCode::ModelLoadFailed);
}

}  // namespace engine
}  // namespace ccinfer
