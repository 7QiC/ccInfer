add_library(engine STATIC
    backend/cuda/cuda_backend.cpp
    backend/cuda/cuda_buffer.cpp
    executor/single_device_executor.cpp
    worker/worker.cpp
    worker/batch_translator.cpp
    kernel/norm/rms_norm.cu
    kernel/pos/rope.cu
    model/rope/rope_cache.cpp
    kernel/mlp/silu_mul.cu
    kernel/attention/naive_attn.cu
    kernel/attention/write_kv_cache.cu
    kernel/attention/prefill_attn.cu
    kernel/attention/decode_attn.cu
    cache/kv_cache_manager.cpp
    cache/kv_cache_storage.cpp
    cache/prefix_cache.cpp
    kernel/elementwise/elementwise.cu
    kernel/sampling/sample.cu
    model/loader.cpp
    model/registry.cpp
    model/qwen3/qwen3_model.cpp
    model/qwen3/qwen3_weights.cpp
)

target_include_directories(engine
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Boost_INCLUDE_DIRS}
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
