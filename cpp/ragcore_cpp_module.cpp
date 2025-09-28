#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr float kEpsilon = 1e-12F;

py::array_t<float, py::array::c_style | py::array::forcecast> ensure_vectors(
    const py::array& vectors, std::size_t dim
) {
    auto converted = py::array_t<float, py::array::c_style | py::array::forcecast>(vectors);
    if (converted.ndim() != 2) {
        throw std::invalid_argument("expected a 2D array of vectors");
    }
    if (static_cast<std::size_t>(converted.shape(1)) != dim) {
        throw std::invalid_argument("vector dimension mismatch");
    }
    return converted;
}

std::optional<py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>> ensure_ids(
    const py::object& ids, std::size_t expected
) {
    if (ids.is_none()) {
        return std::nullopt;
    }
    py::array raw(ids);
    auto converted =
        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>(raw);
    if (converted.ndim() != 1) {
        throw std::invalid_argument("ids must be a 1D array");
    }
    if (static_cast<std::size_t>(converted.shape(0)) != expected) {
        throw std::invalid_argument("ids length must match number of vectors");
    }
    return converted;
}

float compute_distance(
    const float* query,
    const float* vector,
    std::size_t dim,
    const std::string& metric,
    float query_norm
) {
    if (metric == "l2") {
        float sum = 0.0F;
        for (std::size_t i = 0; i < dim; ++i) {
            const float diff = query[i] - vector[i];
            sum += diff * diff;
        }
        return sum;
    }
    if (metric == "ip") {
        float dot = 0.0F;
        for (std::size_t i = 0; i < dim; ++i) {
            dot += query[i] * vector[i];
        }
        return -dot;
    }
    if (metric == "cosine") {
        float dot = 0.0F;
        float vector_norm = 0.0F;
        for (std::size_t i = 0; i < dim; ++i) {
            dot += query[i] * vector[i];
            vector_norm += vector[i] * vector[i];
        }
        vector_norm = std::sqrt(vector_norm);
        const float denom = std::max(query_norm * vector_norm, kEpsilon);
        const float cosine = dot / denom;
        return 1.0F - cosine;
    }
    throw std::invalid_argument("unsupported metric: " + metric);
}

float compute_query_norm(const float* query, std::size_t dim) {
    float norm = 0.0F;
    for (std::size_t i = 0; i < dim; ++i) {
        norm += query[i] * query[i];
    }
    return std::sqrt(norm);
}

}  // namespace

class CPPHandle {
  public:
    CPPHandle(py::dict spec, std::string metric, bool requires_training, bool supports_gpu)
        : spec_dict_(std::move(spec)),
          metric_(std::move(metric)),
          requires_training_(requires_training),
          supports_gpu_(supports_gpu),
          is_trained_(!requires_training),
          is_gpu_(false),
          device_(),
          next_id_(0) {
        dim_ = static_cast<std::size_t>(py::int_(spec_dict_["dim"]));
        kind_ = py::str(spec_dict_["kind"]);
        backend_ = py::str(spec_dict_["backend"]);
    }

    bool requires_training() const { return requires_training_; }

    void train(const py::array& vectors) {
        if (!requires_training_) {
            is_trained_ = true;
            return;
        }
        auto batch = ensure_vectors(vectors, dim_);
        if (batch.shape(0) == 0) {
            throw std::invalid_argument("training vectors cannot be empty");
        }
        is_trained_ = true;
    }

    void add(const py::array& vectors, const py::object& ids = py::none()) {
        if (requires_training_ && !is_trained_) {
            throw std::runtime_error("index requires training before adding vectors");
        }
        auto batch = ensure_vectors(vectors, dim_);
        const std::size_t rows = static_cast<std::size_t>(batch.shape(0));
        if (rows == 0) {
            return;
        }
        auto ids_array = ensure_ids(ids, rows);
        const auto batch_buffer = batch.request();
        const float* data = static_cast<float*>(batch_buffer.ptr);
        std::optional<py::buffer_info> ids_buffer;
        const std::int64_t* id_ptr = nullptr;
        if (ids_array.has_value()) {
            ids_buffer = ids_array->request();
            id_ptr = static_cast<std::int64_t*>(ids_buffer->ptr);
        }
        for (std::size_t row = 0; row < rows; ++row) {
            const float* vector = data + row * dim_;
            vectors_.insert(vectors_.end(), vector, vector + dim_);
            norms_.push_back(compute_query_norm(vector, dim_));
            if (ids_array.has_value()) {
                ids_.push_back(id_ptr[row]);
                next_id_ = std::max(next_id_, ids_.back() + 1);
            } else {
                ids_.push_back(next_id_++);
            }
        }
    }

    py::dict search(const py::array& queries, std::int64_t k) const {
        if (ids_.empty()) {
            throw std::runtime_error("cannot search an empty index");
        }
        auto query_array = ensure_vectors(queries, dim_);
        const std::size_t query_count = static_cast<std::size_t>(query_array.shape(0));
        const std::size_t total = ids_.size();
        const std::size_t limit = static_cast<std::size_t>(std::max<std::int64_t>(0, k));
        const std::size_t topk = std::min(limit, total);

        if (topk == 0) {
            throw std::invalid_argument("k must be greater than zero");
        }

        auto ids_result = py::array_t<std::int64_t>({py::ssize_t(query_count), py::ssize_t(topk)});
        auto distances_result = py::array_t<float>({py::ssize_t(query_count), py::ssize_t(topk)});

        const auto query_buffer = query_array.request();
        const float* query_ptr = static_cast<float*>(query_buffer.ptr);
        auto ids_view = ids_result.mutable_unchecked<2>();
        auto dist_view = distances_result.mutable_unchecked<2>();

        for (std::size_t qi = 0; qi < query_count; ++qi) {
            const float* query_vector = query_ptr + qi * dim_;
            const float query_norm = metric_ == "cosine" ? compute_query_norm(query_vector, dim_) : 0.0F;
            std::vector<std::pair<float, std::size_t>> ranking;
            ranking.reserve(total);
            for (std::size_t vi = 0; vi < total; ++vi) {
                const float* stored_vector = vectors_.data() + vi * dim_;
                float distance = compute_distance(query_vector, stored_vector, dim_, metric_, query_norm);
                ranking.emplace_back(distance, vi);
            }
            std::stable_sort(
                ranking.begin(),
                ranking.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; }
            );
            for (std::size_t rank = 0; rank < topk; ++rank) {
                const auto index = ranking[rank].second;
                ids_view(qi, rank) = ids_[index];
                dist_view(qi, rank) = ranking[rank].first;
            }
        }

        py::dict result;
        result["ids"] = std::move(ids_result);
        result["distances"] = std::move(distances_result);
        return result;
    }

    std::int64_t ntotal() const { return static_cast<std::int64_t>(ids_.size()); }

    py::object serialize_cpu() const {
        py::module interfaces = py::module::import("ragcore.interfaces");
        py::object serialized_index = interfaces.attr("SerializedIndex");

        const std::size_t count = ids_.size();
        py::array_t<float> vectors({py::ssize_t(count), py::ssize_t(dim_)});
        std::memcpy(vectors.mutable_data(), vectors_.data(), sizeof(float) * vectors_.size());

        py::array_t<std::int64_t> ids({py::ssize_t(count)});
        std::memcpy(ids.mutable_data(), ids_.data(), sizeof(std::int64_t) * ids_.size());

        py::dict metadata;
        metadata["requires_training"] = requires_training_;
        metadata["supports_gpu"] = supports_gpu_;

        return serialized_index(
            spec_dict_,
            vectors,
            ids,
            metadata,
            py::bool_(is_trained_),
            py::bool_(is_gpu_)
        );
    }

    std::shared_ptr<CPPHandle> to_gpu(const py::object& device = py::none()) const {
        auto clone = std::make_shared<CPPHandle>(*this);
        if (supports_gpu_) {
            clone->is_gpu_ = true;
            clone->device_ = device.is_none() ? std::string("cuda:0") : device.cast<std::string>();
        }
        return clone;
    }

    std::shared_ptr<CPPHandle> merge_with(const CPPHandle& other) const {
        const int comparison = PyObject_RichCompareBool(other.spec_dict_.ptr(), spec_dict_.ptr(), Py_EQ);
        if (comparison == -1) {
            throw py::error_already_set();
        }
        if (comparison == 0) {
            throw std::invalid_argument("cannot merge handles with different specs");
        }
        auto merged = std::make_shared<CPPHandle>(*this);
        merged->vectors_ = vectors_;
        merged->vectors_.insert(merged->vectors_.end(), other.vectors_.begin(), other.vectors_.end());
        merged->norms_ = norms_;
        merged->norms_.insert(merged->norms_.end(), other.norms_.begin(), other.norms_.end());
        merged->ids_.reserve(ids_.size() + other.ids_.size());
        merged->ids_ = ids_;
        merged->ids_.insert(merged->ids_.end(), other.ids_.begin(), other.ids_.end());
        merged->next_id_ = std::max(next_id_, other.next_id_);
        merged->is_trained_ = is_trained_ || other.is_trained_;
        merged->is_gpu_ = is_gpu_ || other.is_gpu_;
        merged->device_ = is_gpu_ ? device_ : other.device_;
        return merged;
    }

    py::dict spec() const { return py::dict(spec_dict_); }

    bool is_gpu() const { return is_gpu_; }

    py::object device() const {
        if (device_.empty()) {
            return py::none();
        }
        return py::str(device_);
    }

  private:
    py::dict spec_dict_;
    std::string metric_;
    bool requires_training_;
    bool supports_gpu_;
    bool is_trained_;
    bool is_gpu_;
    std::string device_;
    std::vector<float> vectors_;
    std::vector<float> norms_;
    std::vector<std::int64_t> ids_;
    std::size_t dim_;
    std::string kind_;
    std::string backend_;
    std::int64_t next_id_;
};

class CPPBackend {
  public:
    CPPBackend() = default;

    py::dict capabilities() const {
        py::dict kinds;
        py::dict flat_params;
        flat_params["requires_training"] = false;
        kinds["flat"] = flat_params;

        py::list metrics;
        metrics.append("cosine");
        metrics.append("ip");
        metrics.append("l2");

        py::dict result;
        result["name"] = "cpp";
        result["supports_gpu"] = false;
        result["kinds"] = kinds;
        result["metrics"] = metrics;
        return result;
    }

    std::shared_ptr<CPPHandle> build(const py::object& spec) const {
        py::module interfaces = py::module::import("ragcore.interfaces");
        py::object index_spec = interfaces.attr("IndexSpec");
        py::object parsed = index_spec.attr("from_mapping")(spec, py::arg("default_backend") = "cpp");

        std::string backend = py::str(parsed.attr("backend"));
        if (backend != "cpp" && backend != "cpp_faiss") {
            throw std::invalid_argument("CPP backend cannot build other backends");
        }

        std::string kind = py::str(parsed.attr("kind"));
        std::string metric = py::str(parsed.attr("metric"));
        py::dict spec_dict = parsed.attr("as_dict")();

        bool requires_training = kind != "flat";
        return std::make_shared<CPPHandle>(spec_dict, metric, requires_training, false);
    }
};

PYBIND11_MODULE(_ragcore_cpp, m) {
    py::class_<CPPHandle, std::shared_ptr<CPPHandle>>(m, "CPPHandle")
        .def("requires_training", &CPPHandle::requires_training)
        .def("train", &CPPHandle::train, py::arg("vectors"))
        .def("add", &CPPHandle::add, py::arg("vectors"), py::arg("ids") = py::none())
        .def("search", &CPPHandle::search, py::arg("queries"), py::arg("k"))
        .def("ntotal", &CPPHandle::ntotal)
        .def("serialize_cpu", &CPPHandle::serialize_cpu)
        .def("to_gpu", &CPPHandle::to_gpu, py::arg("device") = py::none())
        .def("merge_with", &CPPHandle::merge_with, py::arg("other"))
        .def("spec", &CPPHandle::spec)
        .def_property_readonly("is_gpu", &CPPHandle::is_gpu)
        .def_property_readonly("device", &CPPHandle::device);

    auto backend = py::class_<CPPBackend>(m, "CPPBackend")
                       .def(py::init<>())
                       .def("capabilities", &CPPBackend::capabilities)
                       .def("build", &CPPBackend::build, py::arg("spec"));

    backend.attr("name") = "cpp";

    m.attr("__all__") = py::make_tuple("CPPBackend", "CPPHandle");
}
