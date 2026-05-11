add_executable(test_kv_cache_forward test_kv_cache_forward.cpp)
target_link_libraries(test_kv_cache_forward PRIVATE engine gtest_main CUDA::cudart CUDA::cublas)
gtest_discover_tests(test_kv_cache_forward)

add_executable(test_logits_match test_logits_match.cpp)
target_link_libraries(test_logits_match PRIVATE engine server gtest_main CUDA::cudart CUDA::cublas)
gtest_discover_tests(test_logits_match)

add_executable(test_layer_match test_layer_match.cpp)
target_link_libraries(test_layer_match PRIVATE engine server gtest_main CUDA::cudart CUDA::cublas)
gtest_discover_tests(test_layer_match)

add_executable(test_e2e_server test_e2e_server.cpp)
target_link_libraries(test_e2e_server PRIVATE gtest_main)
target_compile_definitions(test_e2e_server PRIVATE
    CCINFER_SERVER_BINARY="${CMAKE_BINARY_DIR}/src/ccinfer-server")
gtest_discover_tests(test_e2e_server)
