find_package(benchmark CONFIG REQUIRED)

add_executable(strata_bench
  smoke_bench.cpp)
target_link_libraries(strata_bench PRIVATE
  strata::strata strata_options benchmark::benchmark_main)
