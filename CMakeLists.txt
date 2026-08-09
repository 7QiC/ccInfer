cmake_minimum_required(VERSION 3.20)
project(ccInfer VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-std=c++23")
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Boost REQUIRED)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_SERVER "Build server" OFF)

# ---------------------------------------------------------------------------
# Third-party operator library: ccop (https://github.com/7QiC/ccop), pinned as
# a git submodule at third_party/ccop.
#
# On a fresh clone, initialize it with:
#   git submodule update --init --recursive
#
# ccop tests are built inside the ccop project itself (CCop_BUILD_TESTS), not
# in this build tree.
# ---------------------------------------------------------------------------
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ccop/CMakeLists.txt")
    message(FATAL_ERROR
        "ccop submodule missing at third_party/ccop. "
        "Run: git submodule update --init --recursive")
endif()

add_subdirectory(third_party/ccop ccop)

add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()

    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests)
endif()
