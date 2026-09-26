find_package(benchmark CONFIG REQUIRED)

# Microbenchmarks (Google Benchmark).
add_executable(strata_bench
  distance_bench.cpp)
target_link_libraries(strata_bench PRIVATE strata::strata strata_options benchmark::benchmark_main)

# End-to-end search harness, driven by bench/run_search_bench.py.
add_executable(strata_search
  search_harness.cpp)
target_link_libraries(strata_search PRIVATE strata::strata strata_options)
target_compile_definitions(strata_search PRIVATE STRATA_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

# Storage: insert throughput per sync mode, checkpoint and recovery time.
add_executable(strata_storage_bench
  storage_bench.cpp)
target_link_libraries(strata_storage_bench PRIVATE strata::strata strata_options)

# Filtered search: QPS and recall per strategy and selectivity.
add_executable(strata_filter_bench
  filter_bench.cpp)
target_link_libraries(strata_filter_bench PRIVATE strata::strata strata_options)
target_compile_definitions(strata_filter_bench PRIVATE STRATA_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
