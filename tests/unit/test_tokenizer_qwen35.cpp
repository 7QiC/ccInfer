#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "checkpoint/gguf/reader.h"
#include "tokenizer/byte_level_bpe_tokenizer.h"
#include "tokenizer/gguf_tokenizer.h"

namespace ccinfer {
namespace {

#ifndef GGUF_ARTIFACT_PATH
#define GGUF_ARTIFACT_PATH "models/qwen3_5-2B/Qwen3.5-2B-Q8_0.gguf"
#endif
#ifndef TOKENIZER_JSON_PATH
#define TOKENIZER_JSON_PATH "models/qwen3_5-2B/tokenizer.json"
#endif

TEST(TokenizerQwen35Test, GgufAndJsonPathsAgree) {
    const std::string gguf_path = GGUF_ARTIFACT_PATH;
    const std::string json_path = TOKENIZER_JSON_PATH;
    if (!std::filesystem::exists(gguf_path) || !std::filesystem::exists(json_path)) {
        GTEST_SKIP() << "Qwen3.5 model/tokenizer.json not present";
    }

    auto reader_r = GGUFReader::create(gguf_path);
    ASSERT_TRUE(reader_r.has_value());

    auto gguf_tokenizer_r = create_tokenizer_from_gguf(**reader_r);
    ASSERT_TRUE(gguf_tokenizer_r.has_value());
    auto& gguf_tokenizer = **gguf_tokenizer_r;

    ByteLevelBpeTokenizer json_tokenizer;
    auto load_r = json_tokenizer.load(json_path);
    ASSERT_TRUE(load_r.has_value());

    EXPECT_EQ(gguf_tokenizer.vocab_size(), 248320);
    EXPECT_EQ(gguf_tokenizer.vocab_size(), json_tokenizer.vocab_size());
    EXPECT_EQ(gguf_tokenizer.eos_token_id(), json_tokenizer.eos_token_id());
    EXPECT_EQ(gguf_tokenizer.eos_token_id(), 248046);
    EXPECT_EQ(gguf_tokenizer.pad_token_id(), 248055);

    for (const std::string& text : {std::string("Hello"), std::string("Hello world"),
                                    std::string("Qwen3.5 测试"), std::string("你好，世界！")}) {
        auto gguf_ids_r = gguf_tokenizer.encode(text, false);
        ASSERT_TRUE(gguf_ids_r.has_value());
        auto json_ids_r = json_tokenizer.encode(text, false);
        ASSERT_TRUE(json_ids_r.has_value());
        EXPECT_EQ(*gguf_ids_r, *json_ids_r) << "text: " << text;
    }
}

TEST(TokenizerQwen35Test, GenericCreateTokenizerHandlesGgufPath) {
    const std::string gguf_path = GGUF_ARTIFACT_PATH;
    if (!std::filesystem::exists(gguf_path)) {
        GTEST_SKIP() << "Qwen3.5 GGUF artifact not present";
    }

    auto tokenizer_r = create_tokenizer(gguf_path);
    ASSERT_TRUE(tokenizer_r.has_value());
    auto& tokenizer = **tokenizer_r;

    auto reader_r = GGUFReader::create(gguf_path);
    ASSERT_TRUE(reader_r.has_value());
    auto reader_tokenizer_r = create_tokenizer_from_gguf(**reader_r);
    ASSERT_TRUE(reader_tokenizer_r.has_value());
    auto& reader_tokenizer = **reader_tokenizer_r;

    EXPECT_EQ(tokenizer.vocab_size(), reader_tokenizer.vocab_size());
    auto ids_r = tokenizer.encode("Hello world", false);
    ASSERT_TRUE(ids_r.has_value());
    auto reader_ids_r = reader_tokenizer.encode("Hello world", false);
    ASSERT_TRUE(reader_ids_r.has_value());
    EXPECT_EQ(*ids_r, *reader_ids_r);
}

}  // namespace
}  // namespace ccinfer
