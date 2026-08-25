#include <gtest/gtest.h>

#include "backend/backend.h"
#include "model/rope/rope_cache.h"

using namespace ccinfer;

TEST(RopeCacheTest, ClassAndTensorView) {
    auto backend_r = Backend::create(0);
    if (!backend_r) GTEST_SKIP() << "CUDA unavailable";
    auto& backend = **backend_r;
    auto cache_result = RopeCache::create(16, 32, 10000.0f, backend);
    ASSERT_TRUE(cache_result);
    auto& cache = *cache_result;
    EXPECT_EQ(cache.max_position(), 16);
    EXPECT_EQ(cache.rotary_dim(), 32);
    EXPECT_EQ(cache.half_rotary_dim(), 16);
    EXPECT_NE(cache.tensor().data(), nullptr);
    EXPECT_EQ(cache.numel(), 512u);

    const Tensor view = cache.tensor();
    EXPECT_EQ(view.rank(), 3);
    EXPECT_EQ(view.shape(0), 16);
    EXPECT_EQ(view.shape(1), 16);
    EXPECT_EQ(view.shape(2), 2);
    EXPECT_EQ(view.dtype(), ccop::DType::kFloat32);
}
