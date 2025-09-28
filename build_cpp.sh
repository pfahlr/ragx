#!/bin/bash
export pybind11_DIR=$(python -m pybind11 --cmakedir)
cmake -S cpp -B cpp/build && cmake --build cpp/build

