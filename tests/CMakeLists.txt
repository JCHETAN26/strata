find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(strata_tests
  bm25_test.cpp
  brute_force_test.cpp
  bytes_test.cpp
  crash_recovery_test.cpp
  dataset_test.cpp
  distance_test.cpp
  filter_test.cpp
  fusion_test.cpp
  recall_test.cpp
  simd_test.cpp
  smoke_test.cpp
  pq_test.cpp
  storage_test.cpp
  text_test.cpp
  thread_pool_test.cpp)
if(STRATA_HAS_HNSW)
  target_sources(strata_tests PRIVATE hnsw_test.cpp hnsw_persistence_test.cpp)
endif()
target_link_libraries(strata_tests PRIVATE strata::strata strata_options GTest::gtest_main)
# Tests may use internal headers (util/bytes.hpp) as well as the public ones.
target_include_directories(strata_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
# Tests that need downloaded datasets look here and skip if the files are absent.
target_compile_definitions(strata_tests PRIVATE
  STRATA_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
  STRATA_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
gtest_discover_tests(strata_tests)

# Writes C++ results for tests/python/test_bindings.py to compare the Python bindings against.
add_executable(strata_reference strata_reference.cpp)
target_link_libraries(strata_reference PRIVATE strata::strata strata_options)
