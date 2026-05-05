add_executable(test_result unit/test_result.cpp)
target_link_libraries(test_result PRIVATE engine gtest_main)
include(GoogleTest)
gtest_discover_tests(test_result)

add_executable(test_device_buffer unit/test_device_buffer.cpp)
target_link_libraries(test_device_buffer PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_device_buffer)

add_executable(test_tensor unit/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE engine gtest_main)
gtest_discover_tests(test_tensor)

add_executable(test_config unit/test_config.cpp)
target_link_libraries(test_config PRIVATE engine gtest_main)
gtest_discover_tests(test_config)

add_executable(test_smoke unit/test_smoke.cpp)
target_link_libraries(test_smoke PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_smoke)

add_executable(test_rms_norm kernel/test_rms_norm.cu)
target_link_libraries(test_rms_norm PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_rms_norm)

add_executable(test_rope kernel/test_rope.cu)
target_link_libraries(test_rope PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_rope)

add_executable(test_silu_mul kernel/test_silu_mul.cu)
target_link_libraries(test_silu_mul PRIVATE engine gtest_main CUDA::cudart)
gtest_discover_tests(test_silu_mul)

add_executable(test_naive_attn kernel/test_naive_attn.cu)
target_link_libraries(test_naive_attn PRIVATE engine gtest_main CUDA::cudart CUDA::cublas)
gtest_discover_tests(test_naive_attn)

add_executable(test_tokenizer unit/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE engine gtest_main)
gtest_discover_tests(test_tokenizer)

add_executable(test_sampler unit/test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE engine gtest_main)
gtest_discover_tests(test_sampler)

add_executable(test_logits_match integration/test_logits_match.cpp)
target_link_libraries(test_logits_match PRIVATE engine gtest_main CUDA::cudart CUDA::cublas)
gtest_discover_tests(test_logits_match)

add_executable(test_layer_match integration/test_layer_match.cpp)
target_link_libraries(test_layer_match PRIVATE engine gtest_main CUDA::cudart CUDA::cublas)
gtest_discover_tests(test_layer_match)
