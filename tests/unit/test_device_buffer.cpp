#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "engine/backend/cuda/cuda_backend.h"
#include "engine/backend/device_buffer.h"

using namespace ccinfer;
using namespace ccinfer::engine;

TEST(DeviceBufferTest, DefaultConstruction) {
    std::unique_ptr<DeviceBuffer> buf;
    EXPECT_EQ(buf.get(), nullptr);
    EXPECT_TRUE(!buf);
}

TEST(DeviceBufferTest, AllocateAndZero) {
    auto backend_r = CudaBackend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;
    auto buf_r = backend.allocate_buffer(1024 * sizeof(float));
    ASSERT_TRUE(buf_r.has_value());
    auto buf = std::move(*buf_r);
    EXPECT_NE(buf->data(), nullptr);
    EXPECT_FALSE(!buf);
    EXPECT_EQ(buf->bytes() / sizeof(float), 1024);

    cudaMemset(buf->data(), 0, buf->bytes());
    std::vector<float> host(1024, 1.0f);
    cudaMemcpy(host.data(), buf->data(), buf->bytes(), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_EQ(host[i], 0.0f);
    }
}

TEST(DeviceBufferTest, MoveConstruction) {
    auto backend_r = CudaBackend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;
    auto a_r = backend.allocate_buffer(512 * sizeof(float));
    ASSERT_TRUE(a_r.has_value());
    auto a = std::move(*a_r);
    void* ptr = a->data();
    EXPECT_NE(ptr, nullptr);

    auto b = std::move(a);
    EXPECT_EQ(a.get(), nullptr);
    EXPECT_EQ(b->data(), ptr);
    EXPECT_EQ(b->bytes(), 512 * sizeof(float));
}

TEST(DeviceBufferTest, MoveAssignment) {
    auto backend_r = CudaBackend::create(0);
    ASSERT_TRUE(backend_r.has_value());
    auto& backend = **backend_r;
    auto a_r = backend.allocate_buffer(256 * sizeof(float));
    ASSERT_TRUE(a_r.has_value());
    auto a = std::move(*a_r);
    auto b_r = backend.allocate_buffer(128 * sizeof(float));
    ASSERT_TRUE(b_r.has_value());
    auto b = std::move(*b_r);
    void* ptr_a = a->data();
    EXPECT_NE(ptr_a, nullptr);

    b = std::move(a);
    EXPECT_EQ(a.get(), nullptr);
    EXPECT_EQ(b->data(), ptr_a);
    EXPECT_EQ(b->bytes() / sizeof(float), 256);
}
