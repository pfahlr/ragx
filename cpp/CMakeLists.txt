cmake_minimum_required(VERSION 3.15)
project(ragcore_cpp LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_ragcore_cpp src/cpp_backend.cpp)

target_include_directories(_ragcore_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(_ragcore_cpp PRIVATE cxx_std_17)

