add_library(engine STATIC
    backend/cuda/cuda_backend.cpp
    kernel/norm/rms_norm.cu
    kernel/pos/rope.cu
    model/rope/rope_cache.cpp
    kernel/mlp/silu_mul.cu
    kernel/attention/naive_attn.cu
    kernel/elementwise/elementwise.cu
    model/loader.cpp
    model/registry.cpp
    model/llama/llama_model.cpp
    runner/model_runner.cpp
    tokenizer/byte_level_bpe_tokenizer.cpp
    sampling/sampler.cpp
)

target_include_directories(engine
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(engine PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-std=c++23>)

target_link_libraries(engine
    PUBLIC
        nlohmann_json::nlohmann_json
        fmt::fmt
        spdlog::spdlog
    PRIVATE
        CUDA::cublas
        CUDA::cudart
)
