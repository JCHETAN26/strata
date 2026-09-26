# Python extension module strata._core (nanobind). Built by scikit-build-core via
# `pip install -e .`; see pyproject.toml.
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
# scikit-build-core puts nanobind's CMake config on the prefix path; fall back to asking Python.
if(NOT nanobind_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
endif()
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC src/bindings.cpp)
target_link_libraries(_core PRIVATE strata::strata)
target_compile_options(_core PRIVATE -Wall -Wextra)

# Type stubs (strata/_core.pyi) for editors and type checkers.
nanobind_add_stub(_core_stub
  MODULE _core
  OUTPUT _core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_core>
  DEPENDS _core)

install(TARGETS _core LIBRARY DESTINATION strata)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi" DESTINATION strata)
