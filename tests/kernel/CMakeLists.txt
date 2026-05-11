add_executable(test_write_kv_cache test_write_kv_cache.cu)
target_link_libraries(test_write_kv_cache PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_write_kv_cache)

add_executable(test_prefill_attn test_prefill_attn.cu)
target_link_libraries(test_prefill_attn PRIVATE engine gtest_main CUDA::cudart CUDA::cublas)
gtest_discover_tests(test_prefill_attn)

add_executable(test_decode_attn test_decode_attn.cu)
target_link_libraries(test_decode_attn PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_decode_attn)

add_executable(test_rms_norm test_rms_norm.cu)
target_link_libraries(test_rms_norm PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_rms_norm)

add_executable(test_rope test_rope.cu)
target_link_libraries(test_rope PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_rope)

add_executable(test_silu_mul test_silu_mul.cu)
target_link_libraries(test_silu_mul PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_silu_mul)

add_executable(test_naive_attn test_naive_attn.cu)
target_link_libraries(test_naive_attn PRIVATE engine gtest_main CUDA::cudart CUDA::cublas)
gtest_discover_tests(test_naive_attn)

add_executable(test_sample test_sample.cu)
target_link_libraries(test_sample PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_sample)

add_executable(test_gemm test_gemm.cu)
target_link_libraries(test_gemm PRIVATE engine gtest_main CUDA::cudart CUDA::cublas)
gtest_discover_tests(test_gemm)
