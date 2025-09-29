#!/bin/bash
SCRIPT_DIR_ABS="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd "$SCRIPT_DIR_ABS/.."
export pybind11_DIR=$(python -m pybind11 --cmakedir)
cmake -S cpp -B cpp/build && cmake --build cpp/build
cd -
