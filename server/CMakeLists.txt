# gRPC server and sharding (Phase 7). Opt-in: enabled by -DSTRATA_BUILD_SERVER=ON, which the root
# CMakeLists guards this subdirectory behind, so the default build and all existing tests are
# unchanged when gRPC is not installed.
#
# gRPC and Protobuf come from vcpkg (the "server" feature in vcpkg.json) on Linux/CI, or from a
# system install (e.g. Homebrew) on the Mac to avoid building gRPC from source. Both expose the
# same CONFIG packages, so this file is identical either way.

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# --- Generated protobuf + gRPC stubs -------------------------------------------------------------
# Code is generated into the build tree, mirroring the proto's path (strata/v1/...), so includes
# read "strata/v1/vector_service.grpc.pb.h".
set(STRATA_PROTO_ROOT "${PROJECT_SOURCE_DIR}/proto")
set(STRATA_PROTO_FILE "${STRATA_PROTO_ROOT}/strata/v1/vector_service.proto")

add_library(strata_proto STATIC "${STRATA_PROTO_FILE}")
target_link_libraries(strata_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
target_include_directories(strata_proto PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
# Generated sources are not ours to lint: this target does not link strata_options, so the
# project's strict warnings never apply to them. Position-independent so it can be linked into
# shared objects too.
set_target_properties(strata_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)

protobuf_generate(
  TARGET strata_proto
  IMPORT_DIRS "${STRATA_PROTO_ROOT}"
  PROTOC_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")

protobuf_generate(
  TARGET strata_proto
  LANGUAGE grpc
  GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
  PLUGIN "protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
  IMPORT_DIRS "${STRATA_PROTO_ROOT}"
  PROTOC_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# --- Server library (shard + coordinator services) -----------------------------------------------
add_library(strata_server STATIC
  shard_service.cpp
  coordinator_service.cpp)
target_link_libraries(strata_server
  PUBLIC strata::strata strata_proto gRPC::grpc++
  PRIVATE strata_options)
# Server headers are included by name ("shard_service.hpp") from this directory only, and the
# generated stubs as "strata/v1/..." from strata_proto's PUBLIC include dir. (Exposing the whole
# repository root would let any consumer include any file in it.)
target_include_directories(strata_server PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# --- Executables ---------------------------------------------------------------------------------
add_executable(strata_shard strata_shard.cpp)
target_link_libraries(strata_shard PRIVATE strata_server strata_options)

add_executable(strata_coordinator strata_coordinator.cpp)
target_link_libraries(strata_coordinator PRIVATE strata_server strata_options)

# --- Tests ---------------------------------------------------------------------------------------
if(STRATA_BUILD_TESTS)
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)
  add_executable(strata_server_tests
    tests/id_codec_test.cpp
    tests/sharding_test.cpp)
  target_link_libraries(strata_server_tests PRIVATE strata_server strata_options GTest::gtest_main)
  if(STRATA_SANITIZE)
    # protobuf annotates RepeatedField storage for ASan's container-overflow check. When protobuf
    # itself is not ASan-built (Homebrew, default vcpkg triplets), the annotations are only half
    # applied and reads of request->values() report false container overflows. Turn that one
    # check off for these tests only; the core test suite keeps it. Everything else ASan and
    # UBSan check stays on.
    gtest_discover_tests(strata_server_tests PROPERTIES ENVIRONMENT
      "ASAN_OPTIONS=abort_on_error=1:detect_container_overflow=0")
  else()
    gtest_discover_tests(strata_server_tests)
  endif()
endif()
