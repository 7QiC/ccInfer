#include <gtest/gtest.h>
#include "result.h"

TEST(ResultTest, Success) {
    Result<int> r = 42;
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 42);
    EXPECT_EQ(*r, 42);
}

TEST(ResultTest, Error) {
    Result<int> r = std::unexpected(ErrorCode::CudaOutOfMemory);
    EXPECT_FALSE(r.has_value());
    EXPECT_EQ(r.error(), ErrorCode::CudaOutOfMemory);
}

TEST(ResultTest, AndThenChaining) {
    auto f = [](int x) -> Result<int> { return x * 2; };
    Result<int> ok = 21;
    EXPECT_EQ(ok.and_then(f).value(), 42);

    Result<int> err = std::unexpected(ErrorCode::ModelLoadFailed);
    EXPECT_FALSE(err.and_then(f).has_value());
}

TEST(ResultTest, OrElseRecovery) {
    Result<int> err = std::unexpected(ErrorCode::CudaOutOfMemory);
    auto recovered = err.or_else([](ErrorCode) -> Result<int> {
        return 0;
    });
    EXPECT_EQ(recovered.value(), 0);
}

TEST(ResultTest, ErrorMessage) {
    EXPECT_STREQ(error_message(ErrorCode::KVBlockExhausted).data(), "KV cache blocks exhausted");
    EXPECT_STREQ(error_message(ErrorCode::CudaOutOfMemory).data(), "CUDA out of memory");
}
