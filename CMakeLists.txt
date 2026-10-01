cmake_minimum_required(VERSION 3.25)

# Use vcpkg from VCPKG_ROOT when no toolchain was given (e.g. pip / scikit-build-core builds).
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "")
endif()

# Python builds on Linux (pip / uv / scikit-build-core) use GCC 13 when it is installed, the same
# compiler as the linux-* presets, so the extension and the C++ tests come from one compiler.
# A machine without /usr/bin/g++-13 (e.g. Kaggle) keeps its default; macOS is unaffected. The CXX
# environment variable is not honored here: build frontends set it from Python's sysconfig
# ("c++"), so it does not mean the user chose a compiler. To choose one, pass
# -C cmake.define.CMAKE_CXX_COMPILER=... (pip) or --config-settings (uv).
if(STRATA_BUILD_PYTHON AND CMAKE_HOST_LINUX AND NOT DEFINED CMAKE_CXX_COMPILER
   AND EXISTS "/usr/bin/g++-13")
  set(CMAKE_CXX_COMPILER "/usr/bin/g++-13")
endif()

project(strata VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(STRATA_BUILD_TESTS "Build unit tests" ON)
option(STRATA_BUILD_BENCHMARKS "Build benchmarks" ON)
option(STRATA_BUILD_PYTHON "Build the Python extension module (strata._core)" OFF)
option(STRATA_BUILD_SERVER "Build the gRPC server and sharding coordinator (needs gRPC + Protobuf)" OFF)
option(STRATA_SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(STRATA_TSAN "Build with ThreadSanitizer (exclusive with STRATA_SANITIZE)" OFF)

# SIMD kernels are selected at compile time. NEON is baseline on arm64; AVX2 + FMA must be enabled
# explicitly on x86_64 (on by default there; turn off for machines without AVX2).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64" OR CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
  set(STRATA_IS_X86_64 ON)
endif()
include(CMakeDependentOption)
cmake_dependent_option(STRATA_AVX2 "Compile AVX2 + FMA kernels (x86_64)" ON "STRATA_IS_X86_64" OFF)

# Floating point: no FMA contraction of separate multiply and add, no fast-math. The scalar kernel
# is the reference, so it must compute the same IEEE operations with every compiler (clang
# contracts by default, GCC does not in -std=c++20 mode). Explicit intrinsics (vfmaq, fmadd) are
# unaffected. Every target that computes distances gets these flags, and build_info.hpp records
# them so results from two builds can be compared bit for bit only when they match.
set(STRATA_FP_FLAGS -ffp-contract=off -fno-fast-math)

# Warnings and sanitizers apply to every Strata target through this interface library.
add_library(strata_options INTERFACE)
target_compile_options(strata_options INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
if(STRATA_SANITIZE)
  target_compile_options(strata_options INTERFACE
    -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
  target_link_options(strata_options INTERFACE -fsanitize=address,undefined)
endif()
if(STRATA_TSAN)
  target_compile_options(strata_options INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
  target_link_options(strata_options INTERFACE -fsanitize=thread)
endif()

find_package(tl-expected CONFIG REQUIRED)
find_package(utf8proc CONFIG REQUIRED)

add_library(strata STATIC
  src/distance/avx2.cpp
  src/distance/dispatch.cpp
  src/distance/neon.cpp
  src/distance/scalar.cpp
  src/eval/recall.cpp
  src/filter/filter.cpp
  src/index/brute_force.cpp
  src/index/fusion.cpp
  src/pq/kmeans.cpp
  src/pq/pq.cpp
  src/storage/collection.cpp
  src/storage/crc32c.cpp
  src/storage/dataset.cpp
  src/storage/file.cpp
  src/storage/snapshot.cpp
  src/storage/wal.cpp
  src/text/analyzer.cpp
  src/text/bm25.cpp
  src/text/porter.cpp
  src/util/thread_pool.cpp)
add_library(strata::strata ALIAS strata)

# HNSW is part of the library (Collection depends on it). STRATA_HAS_HNSW stays defined, and the
# CMake variable stays ON, for code that was written when HNSW was optional.
set(STRATA_HAS_HNSW ON)
target_sources(strata PRIVATE src/index/hnsw.cpp)
target_compile_definitions(strata PUBLIC STRATA_HAS_HNSW=1)
target_include_directories(strata PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
# Internal headers shared across src/ subdirectories (e.g. util/bytes.hpp); not exported.
target_include_directories(strata PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(strata PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(strata
  PUBLIC tl::expected Threads::Threads
  PRIVATE strata_options utf8proc::utf8proc)
# PUBLIC so tests, tools, benchmarks, and the Python module compile with the same FP semantics.
target_compile_options(strata PUBLIC ${STRATA_FP_FLAGS})
if(STRATA_BUILD_PYTHON)
  set_target_properties(strata PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
if(STRATA_AVX2)
  # PUBLIC: headers test __AVX2__ to declare the avx2:: kernels, so every consumer needs the flags.
  target_compile_options(strata PUBLIC -mavx2 -mfma)
  list(APPEND STRATA_FP_FLAGS -mavx2 -mfma)
endif()

# Generated build description (compiler, FP flags, target) for result provenance.
list(JOIN STRATA_FP_FLAGS " " STRATA_FP_FLAGS_STRING)
if(CMAKE_OSX_ARCHITECTURES)
  set(STRATA_TARGET_ARCH "${CMAKE_OSX_ARCHITECTURES}")
else()
  set(STRATA_TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()
configure_file(include/strata/build_info.hpp.in
               "${CMAKE_CURRENT_BINARY_DIR}/generated/strata/build_info.hpp" @ONLY)
target_include_directories(strata PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>)

if(STRATA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(STRATA_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

if(STRATA_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(STRATA_BUILD_SERVER)
  if(STRATA_TSAN)
    # protobuf adds a TSan-only member to every message when compiled with -fsanitize=thread, so
    # our generated code and a non-TSan libprotobuf disagree on message layout and crash. TSan on
    # the server needs gRPC and protobuf built with TSan too (a vcpkg sanitizer triplet).
    message(WARNING "STRATA_TSAN with STRATA_BUILD_SERVER requires gRPC and protobuf built with "
                    "ThreadSanitizer; against stock libraries the server tests crash.")
  endif()
  add_subdirectory(server)
endif()
