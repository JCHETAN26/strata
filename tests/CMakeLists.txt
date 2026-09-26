find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(strata_tests
  smoke_test.cpp)
target_link_libraries(strata_tests PRIVATE
  strata::strata strata_options GTest::gtest_main)
gtest_discover_tests(strata_tests)
