# Test entry points. C++ builds use the CMake presets directly (see README).
#
# Python tests run with PYTHONPATH removed and pytest plugin autoloading disabled, so whatever a
# shell or another project exported (PYTHONPATH entries, pytest plugins installed in the
# environment) cannot change which modules are imported or how tests run. Imports come only
# from the package and the pythonpath list in pyproject.toml.

PYTEST := env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
PRESET ?= debug

.PHONY: test test-cpp test-python

test: test-cpp test-python

test-cpp:
	cmake --build --preset $(PRESET)
	ctest --preset $(PRESET)

test-python:
	$(PYTEST) $(PYTEST_ARGS)
