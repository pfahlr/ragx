cmake_minimum_required(VERSION 3.15)
project(ragcore_cpp_stub LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_ragcore_cpp MODULE ragcore_cpp_module.cpp)
target_compile_features(_ragcore_cpp PRIVATE cxx_std_17)

target_include_directories(
    _ragcore_cpp
    PRIVATE
        ${Python_INCLUDE_DIRS}
)
