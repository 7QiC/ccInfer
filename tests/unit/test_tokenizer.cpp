#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "engine/tokenizer/byte_level_bpe_tokenizer.h"

namespace {

const char* kMinimalTokenizerJson = R"JSON({
  "model": {
    "type": "BPE",
    "vocab": {
      "<|endoftext|>": 0,
      "!": 1,
      "\"": 2,
      "#": 3,
      "$": 4,
      "%": 5,
      "&": 6,
      "'": 7,
      "(": 8,
      ")": 9,
      "*": 10,
      "+": 11,
      ",": 12,
      "-": 13,
      ".": 14,
      "/": 15,
      "0": 16,
      "1": 17,
      "2": 18,
      "3": 19,
      "4": 20,
      "5": 21,
      "6": 22,
      "7": 23,
      "8": 24,
      "9": 25,
      ":": 26,
      ";": 27,
      "<": 28,
      "=": 29,
      ">": 30,
      "?": 31,
      "@": 32,
      "A": 33,
      "B": 34,
      "C": 35,
      "D": 36,
      "E": 37,
      "F": 38,
      "G": 39,
      "H": 40,
      "I": 41,
      "J": 42,
      "K": 43,
      "L": 44,
      "M": 45,
      "N": 46,
      "O": 47,
      "P": 48,
      "Q": 49,
      "R": 50,
      "S": 51,
      "T": 52,
      "U": 53,
      "V": 54,
      "W": 55,
      "X": 56,
      "Y": 57,
      "Z": 58,
      "[": 59,
      "\\": 60,
      "]": 61,
      "^": 62,
      "_": 63,
      "`": 64,
      "a": 65,
      "b": 66,
      "c": 67,
      "d": 68,
      "e": 69,
      "f": 70,
      "g": 71,
      "h": 72,
      "i": 73,
      "j": 74,
      "k": 75,
      "l": 76,
      "m": 77,
      "n": 78,
      "o": 79,
      "p": 80,
      "q": 81,
      "r": 82,
      "s": 83,
      "t": 84,
      "u": 85,
      "v": 86,
      "w": 87,
      "x": 88,
      "y": 89,
      "z": 90,
      "{": 91,
      "|": 92,
      "}": 93,
      "~": 94,
      " ": 95,
      "\t": 96,
      "\n": 97,
      "Hello": 98,
      " world": 99
    }
  }
})JSON";

}  // namespace

class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        path_ = "/tmp/test_tokenizer_minimal.json";
        std::ofstream f(path_);
        f << kMinimalTokenizerJson;
        f.close();
    }

    void TearDown() override { std::remove(path_.c_str()); }

    std::string path_;
};

TEST_F(TokenizerTest, LoadAndVocabSize) {
    ccinfer::engine::ByteLevelBpeTokenizer tok;
    ASSERT_TRUE(tok.load(path_));
    EXPECT_EQ(tok.vocab_size(), 100);
}

TEST_F(TokenizerTest, EncodeSingleChar) {
    ccinfer::engine::ByteLevelBpeTokenizer tok;
    ASSERT_TRUE(tok.load(path_));

    auto ids = tok.encode("H");
    ASSERT_TRUE(ids);
    ASSERT_EQ(ids->size(), 1);
    EXPECT_EQ((*ids)[0], 40);
}

TEST_F(TokenizerTest, EncodeMultiChar) {
    ccinfer::engine::ByteLevelBpeTokenizer tok;
    ASSERT_TRUE(tok.load(path_));

    auto ids = tok.encode("Hello");
    ASSERT_TRUE(ids);
    ASSERT_EQ(ids->size(), 5);
    EXPECT_EQ((*ids)[0], 40);
    EXPECT_EQ((*ids)[1], 69);
    EXPECT_EQ((*ids)[2], 76);
    EXPECT_EQ((*ids)[3], 76);
    EXPECT_EQ((*ids)[4], 79);
}

TEST_F(TokenizerTest, DecodeSingleToken) {
    ccinfer::engine::ByteLevelBpeTokenizer tok;
    ASSERT_TRUE(tok.load(path_));

    auto text = tok.decode({40});
    ASSERT_TRUE(text);
    EXPECT_EQ(*text, "H");

    auto text2 = tok.decode({79});
    ASSERT_TRUE(text2);
    EXPECT_EQ(*text2, "o");
}

TEST_F(TokenizerTest, EncodeDecodeRoundtrip) {
    ccinfer::engine::ByteLevelBpeTokenizer tok;
    ASSERT_TRUE(tok.load(path_));

    auto ids = tok.encode("Hello");
    ASSERT_TRUE(ids);

    auto text = tok.decode(*ids);
    ASSERT_TRUE(text);
    EXPECT_EQ(*text, "Hello");
}
