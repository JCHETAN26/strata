find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(strata_tests
  brute_force_test.cpp
  dataset_test.cpp
  distance_test.cpp
  recall_test.cpp
  smoke_test.cpp)
if(STRATA_HAS_HNSW)
  target_sources(strata_tests PRIVATE hnsw_test.cpp)
endif()
target_link_libraries(strata_tests PRIVATE strata::strata strata_options GTest::gtest_main)
# Tests that need downloaded datasets look here and skip if the files are absent.
target_compile_definitions(strata_tests PRIVATE
  STRATA_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
gtest_discover_tests(strata_tests)
