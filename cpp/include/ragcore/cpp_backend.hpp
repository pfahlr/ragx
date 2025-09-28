#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

namespace ragcore {

class CppHandle {
  public:
    explicit CppHandle(pybind11::dict spec);

    bool requires_training() const;
    void train(const pybind11::array_t<float>& vectors);
    void add(const pybind11::array_t<float>& vectors, pybind11::object ids = pybind11::none());
    pybind11::dict search(const pybind11::array_t<float>& queries, std::size_t k) const;
    std::size_t ntotal() const;
    pybind11::dict serialize_cpu() const;
    CppHandle to_gpu(pybind11::object device) const;
    CppHandle merge_with(const CppHandle& other) const;
    pybind11::dict spec() const;

  private:
    pybind11::dict spec_;
    std::size_t dim_ = 0;
    std::size_t count_ = 0;
    bool trained_ = false;
    std::vector<float> storage_;
    std::vector<std::int64_t> ids_;
    std::string metric_;
};

class CppBackend {
  public:
    CppBackend() = default;
    pybind11::dict capabilities() const;
    CppHandle build(pybind11::dict spec) const;
};

void bind_cpp_backend(pybind11::module_& module);

}  // namespace ragcore
