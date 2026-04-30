add_library(engine STATIC
    backend/cuda/cuda_backend.cpp
    kernel/norm/rms_norm.cu
    kernel/pos/rope.cu
    kernel/pos/rope_cache.cpp
)

target_include_directories(engine
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(engine
    PUBLIC
        nlohmann_json::nlohmann_json
        fmt::fmt
        spdlog::spdlog
    PRIVATE
        CUDA::cublas
        CUDA::cudart
)
