# Set before project() for CUDA architecture detection
set(CMAKE_CUDA_ARCHITECTURES 89)

cmake_minimum_required(VERSION 3.20)
project(ccInfer VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Boost REQUIRED)

# GTest via FetchContent
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/v1.15.2.tar.gz
)
FetchContent_MakeAvailable(googletest)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_SERVER "Build server" OFF)

add_subdirectory(src/engine)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
