#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "core/device_buffer.h"

using namespace ccinfer;
using namespace ccinfer::engine;

TEST(DeviceBufferTest, DefaultConstruction) {
    DeviceBuffer<float> buf;
    EXPECT_EQ(buf.get(), nullptr);
    EXPECT_TRUE(buf.empty());
}

TEST(DeviceBufferTest, AllocateAndZero) {
    DeviceBuffer<float> buf(1024);
    EXPECT_NE(buf.get(), nullptr);
    EXPECT_FALSE(buf.empty());
    EXPECT_EQ(buf.size(), 1024);

    cudaMemset(buf.get(), 0, 1024 * sizeof(float));
    std::vector<float> host(1024, 1.0f);
    cudaMemcpy(host.data(), buf.get(), 1024 * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < 1024; i++) {
        EXPECT_EQ(host[i], 0.0f);
    }
}

TEST(DeviceBufferTest, MoveConstruction) {
    DeviceBuffer<float> a(512);
    float* ptr = a.get();
    DeviceBuffer<float> b = std::move(a);
    EXPECT_EQ(a.get(), nullptr);
    EXPECT_TRUE(a.empty());
    EXPECT_EQ(b.get(), ptr);
    EXPECT_EQ(b.size(), 512);
}

TEST(DeviceBufferTest, MoveAssignment) {
    DeviceBuffer<float> a(256);
    DeviceBuffer<float> b(128);
    float* ptr_a = a.get();
    b = std::move(a);
    EXPECT_EQ(a.get(), nullptr);
    EXPECT_EQ(b.get(), ptr_a);
    EXPECT_EQ(b.size(), 256);
}
