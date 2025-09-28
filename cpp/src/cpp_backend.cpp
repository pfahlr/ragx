#include "ragcore/cpp_backend.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>

#include <string>

namespace ragcore {

using pybind11::array_t;
using pybind11::dict;
using pybind11::list;
using pybind11::module_;
using pybind11::object;

namespace {
std::size_t infer_dim(const dict& spec) {
    if (spec.contains("dim")) {
        try {
            return spec["dim"].cast<std::size_t>();
        } catch (const pybind11::cast_error&) {
            return 0;
        }
    }
    return 0;
}

std::string infer_metric(const dict& spec) {
    if (spec.contains("metric")) {
        try {
            return spec["metric"].cast<std::string>();
        } catch (const pybind11::cast_error&) {
            return "ip";
        }
    }
    return "ip";
}
}

CppHandle::CppHandle(dict spec) : spec_(std::move(spec)), dim_(infer_dim(spec_)), metric_(infer_metric(spec_)) {}

bool CppHandle::requires_training() const { return false; }

void CppHandle::train(const array_t<float>& /*vectors*/) { trained_ = true; }

void CppHandle::add(const array_t<float>& vectors, object /*ids*/) {
    auto buffer = vectors.request();
    if (buffer.ndim != 2) {
        throw pybind11::value_error("expected a 2D float32 array");
    }
    auto rows = static_cast<std::size_t>(buffer.shape[0]);
    auto cols = static_cast<std::size_t>(buffer.shape[1]);
    if (dim_ == 0) {
        dim_ = cols;
    }
    if (cols != dim_) {
        throw pybind11::value_error("vector dimension mismatch for cpp handle");
    }

    auto view = vectors.unchecked<2>();
    storage_.reserve(storage_.size() + rows * cols);
    ids_.reserve(ids_.size() + rows);

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            storage_.push_back(view(row, col));
        }
        ids_.push_back(static_cast<std::int64_t>(ids_.size()));
    }
    count_ = ids_.size();
}

dict CppHandle::search(const array_t<float>& queries, std::size_t k) const {
    auto qbuf = queries.request();
    if (qbuf.ndim != 2) {
        throw pybind11::value_error("expected a 2D float32 query array");
    }
    if (count_ == 0) {
        throw pybind11::value_error("cannot search an empty index");
    }
    if (static_cast<std::size_t>(qbuf.shape[1]) != dim_) {
        throw pybind11::value_error("query dimension mismatch for cpp handle");
    }
    auto rows = static_cast<std::size_t>(qbuf.shape[0]);
    auto actual_k = std::min<std::size_t>(k, count_);

    array_t<std::int64_t> ids(pybind11::array::ShapeContainer{static_cast<pybind11::ssize_t>(rows), static_cast<pybind11::ssize_t>(actual_k)});
    array_t<float> distances(pybind11::array::ShapeContainer{static_cast<pybind11::ssize_t>(rows), static_cast<pybind11::ssize_t>(actual_k)});

    auto ids_view = ids.mutable_unchecked<2>();
    auto dist_view = distances.mutable_unchecked<2>();
    auto qview = queries.unchecked<2>();

    std::vector<std::pair<float, std::int64_t>> workspace(ids_.size());

    const bool use_l2 = metric_ == "l2";
    const bool use_ip = metric_ == "ip";
    if (!use_l2 && !use_ip) {
        throw pybind11::value_error("unsupported metric for cpp handle: " + metric_);
    }

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t idx = 0; idx < ids_.size(); ++idx) {
            const float* base_ptr = storage_.data() + (idx * dim_);
            const float* query_ptr = &qview(row, 0);
            float distance = 0.0f;
            if (use_l2) {
                for (std::size_t col = 0; col < dim_; ++col) {
                    const float diff = query_ptr[col] - base_ptr[col];
                    distance += diff * diff;
                }
            } else {  // ip
                for (std::size_t col = 0; col < dim_; ++col) {
                    distance -= query_ptr[col] * base_ptr[col];
                }
            }
            workspace[idx] = {distance, ids_[idx]};
        }

        std::stable_sort(workspace.begin(), workspace.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first < rhs.first) {
                return true;
            }
            if (lhs.first > rhs.first) {
                return false;
            }
            return lhs.second < rhs.second;
        });

        for (std::size_t j = 0; j < actual_k; ++j) {
            ids_view(row, j) = workspace[j].second;
            dist_view(row, j) = workspace[j].first;
        }
    }

    dict result;
    result["ids"] = std::move(ids);
    result["distances"] = std::move(distances);
    return result;
}

std::size_t CppHandle::ntotal() const { return ids_.size(); }

dict CppHandle::serialize_cpu() const {
    dict payload;
    payload["spec"] = spec_;
    payload["metadata"] = dict();
    payload["is_trained"] = trained_;
    payload["is_gpu"] = false;

    if (dim_ == 0 || ids_.empty()) {
        payload["vectors"] = array_t<float>(pybind11::array::ShapeContainer{0, 0});
        payload["ids"] = array_t<std::int64_t>(pybind11::array::ShapeContainer{0});
        return payload;
    }

    array_t<float> vectors({static_cast<pybind11::ssize_t>(ids_.size()), static_cast<pybind11::ssize_t>(dim_)});
    array_t<std::int64_t> ids({static_cast<pybind11::ssize_t>(ids_.size())});

    auto vec_view = vectors.mutable_unchecked<2>();
    for (std::size_t row = 0; row < ids_.size(); ++row) {
        for (std::size_t col = 0; col < dim_; ++col) {
            vec_view(row, col) = storage_[row * dim_ + col];
        }
    }

    auto ids_view = ids.mutable_unchecked<1>();
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        ids_view(i) = ids_[i];
    }

    payload["vectors"] = std::move(vectors);
    payload["ids"] = std::move(ids);
    return payload;
}

CppHandle CppHandle::to_gpu(object /*device*/) const { return *this; }

CppHandle CppHandle::merge_with(const CppHandle& other) const {
    if (dim_ != other.dim_) {
        throw pybind11::value_error("cannot merge cpp handles with different dimensions");
    }
    CppHandle merged(spec_);
    merged.dim_ = dim_;
    merged.storage_ = storage_;
    merged.storage_.insert(merged.storage_.end(), other.storage_.begin(), other.storage_.end());
    merged.ids_.resize(ids_.size() + other.ids_.size());
    std::iota(merged.ids_.begin(), merged.ids_.end(), 0);
    merged.count_ = merged.ids_.size();
    merged.trained_ = trained_ || other.trained_;
    return merged;
}

dict CppHandle::spec() const { return spec_; }

dict CppBackend::capabilities() const {
    dict caps;
    caps["name"] = "cpp";
    caps["available"] = true;
    dict kinds;
    dict flat;
    flat["requires_training"] = false;
    list metrics;
    metrics.append(pybind11::str("l2"));
    metrics.append(pybind11::str("ip"));
    flat["metrics"] = std::move(metrics);
    kinds["flat"] = std::move(flat);
    caps["kinds"] = std::move(kinds);
    caps["supports_gpu"] = false;
    caps["notes"] = "stub cpp backend";
    return caps;
}

CppHandle CppBackend::build(dict spec) const { return CppHandle(std::move(spec)); }

void bind_cpp_backend(module_& module) {
    pybind11::class_<CppHandle>(module, "CppHandle")
        .def(pybind11::init<dict>())
        .def("requires_training", &CppHandle::requires_training)
        .def("train", &CppHandle::train)
        .def("add", &CppHandle::add, pybind11::arg("vectors"), pybind11::arg("ids") = pybind11::none())
        .def("search", &CppHandle::search)
        .def("ntotal", &CppHandle::ntotal)
        .def("serialize_cpu", &CppHandle::serialize_cpu)
        .def("to_gpu", &CppHandle::to_gpu, pybind11::arg("device") = pybind11::none())
        .def("merge_with", &CppHandle::merge_with)
        .def("spec", &CppHandle::spec);

    pybind11::class_<CppBackend>(module, "CppBackend")
        .def(pybind11::init<>())
        .def("capabilities", &CppBackend::capabilities)
        .def("build", &CppBackend::build);
}

}  // namespace ragcore

PYBIND11_MODULE(_ragcore_cpp, module) {
    module.doc() = "RAGCore stub C++ backend";
    ragcore::bind_cpp_backend(module);
}
