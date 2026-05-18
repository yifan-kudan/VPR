#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <filesystem>
#include <memory>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>

#include "DBoW2.h"

namespace py = pybind11;

// numpy to feature convertor template
template <int DescDim>
struct FloatDescriptorTraits {
    using NumpyArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
    using FeatureSet = std::vector<std::vector<float>>;

    static FeatureSet from_numpy(NumpyArray descriptors) {
        auto buf = descriptors.request();
        if (buf.ndim != 2 || buf.shape[1] != DescDim) {
            throw std::runtime_error(
                "Input should be a 2D array with shape (N, " + std::to_string(DescDim) + ")");
        }

        auto *ptr = static_cast<float *>(buf.ptr);
        int rows = static_cast<int>(buf.shape[0]);
        FeatureSet features;
        features.reserve(rows);

        for (int i = 0; i < rows; ++i) {
            features.emplace_back(ptr + i * DescDim, ptr + (i + 1) * DescDim);
        }

        return features;
    }
};

struct SiftTraits : FloatDescriptorTraits<128> {
    using Vocabulary = SiftVocabulary;
    using Database = SiftDatabase;

    static Vocabulary make_vocabulary(int k, int L) {
        return Vocabulary(k, L, DBoW2::TF_IDF, DBoW2::L2_NORM);
    }
};

struct SuperpointTraits : FloatDescriptorTraits<256> {
    using Vocabulary = SuperpointVocabulary;
    using Database = SuperpointDatabase;

    static Vocabulary make_vocabulary(int k, int L) {
        return Vocabulary(k, L, DBoW2::TF_IDF, DBoW2::L2_NORM);
    }
};

// ORB descriptors: (N, 32) uint8 -> std::vector<cv::Mat>
struct OrbTraits {
    using NumpyArray = py::array_t<unsigned char, py::array::c_style | py::array::forcecast>;
    using Vocabulary = OrbVocabulary;
    using Database = OrbDatabase;
    using FeatureSet = std::vector<cv::Mat>;

    static Vocabulary make_vocabulary(int k, int L) {
        return Vocabulary(k, L, DBoW2::TF_IDF, DBoW2::L1_NORM);
    }

    static FeatureSet from_numpy(NumpyArray descriptors) {
        auto buf = descriptors.request();

        // ORB descriptors should have 32 columns, each unit of the descriptor is 8 bits,
        // total 256 bits per descriptor. The OpenCV ORB descriptors shape is (N, 32),
        // N is keypoints number in the image. Each row corresponds to a descriptor.
        if (buf.ndim != 2 || buf.shape[1] != 32) {
            throw std::runtime_error("Input should be a 2D array with shape (N, 32)");
        }

        auto *ptr = static_cast<unsigned char *>(buf.ptr);
        int rows = static_cast<int>(buf.shape[0]);
        int cols = static_cast<int>(buf.shape[1]);

        cv::Mat mat(rows, cols, CV_8U, ptr);
        FeatureSet features;
        features.reserve(rows);

        for (int i = 0; i < rows; ++i) {
            features.push_back(mat.row(i).clone());
        }

        return features;
    }
};

template <class Traits>
class PyDBoW2Database {
public:
    PyDBoW2Database(int k, int L)
        : vocabulary_(Traits::make_vocabulary(k, L)), database_(nullptr) {}

    void load_vocabulary(const std::filesystem::path &path) {
        typename Traits::Vocabulary voc;

        try {
            if (path.extension() == ".txt") {
                voc.loadFromTextFile(path.string());
            } else {
                voc.load(path.string());
            }
        } catch (const std::string &e) {
            throw std::runtime_error("Failed to load vocabulary: " + e);
        } catch (const cv::Exception &e) {
            throw std::runtime_error("Failed to load vocabulary: " + std::string(e.what()));
        }

        if (voc.empty()) {
            throw std::runtime_error("Loaded vocabulary is empty: " + path.string());
        }

        vocabulary_ = std::move(voc);
        database_ = std::make_unique<typename Traits::Database>(vocabulary_, false, 0);
    }

    void create_vocabulary(
        const std::vector<typename Traits::NumpyArray> &descriptors_list) {
        std::vector<typename Traits::FeatureSet> all_features;
        all_features.reserve(descriptors_list.size());

        for (const auto &descriptors : descriptors_list) {
            all_features.push_back(Traits::from_numpy(descriptors));
        }

        vocabulary_.create(all_features);
        database_ = std::make_unique<typename Traits::Database>(vocabulary_, false, 0);
    }

    int add(typename Traits::NumpyArray descriptors) {
        if (!database_) {
            throw std::runtime_error("Vocabulary must be created before adding features to the database.");
        }

        return database_->add(Traits::from_numpy(descriptors));
    }

    std::vector<std::pair<int, double>> query(
        typename Traits::NumpyArray descriptors, int top_k = 1) {
        if (!database_) {
            throw std::runtime_error("Vocabulary must be created before querying the database.");
        }

        DBoW2::QueryResults results;
        database_->query(Traits::from_numpy(descriptors), results, top_k);
        std::vector<std::pair<int, double>> output;

        for (const auto &result : results) {
            output.emplace_back(result.Id, result.Score);
        }

        return output;
    }

    void save_database(const std::filesystem::path &path) {
        if (!database_) {
            throw std::runtime_error("Vocabulary must be created before saving the database.");
        }

        database_->save(path.string());
    }

private:
    typename Traits::Vocabulary vocabulary_;
    std::unique_ptr<typename Traits::Database> database_;
};

using PyOrbDatabase = PyDBoW2Database<OrbTraits>;
using PySiftDatabase = PyDBoW2Database<SiftTraits>;
using PySuperpointDatabase = PyDBoW2Database<SuperpointTraits>;

PYBIND11_MODULE(dbow2_cpp, m) {
    py::class_<PyOrbDatabase>(m, "OrbDatabase")
        .def(py::init<int, int>(), py::arg("k") = 9, py::arg("L") = 3)
        .def("load_vocabulary", &PyOrbDatabase::load_vocabulary, py::arg("path"))
        .def("create_vocabulary", &PyOrbDatabase::create_vocabulary, py::arg("descriptors_list"))
        .def("add", &PyOrbDatabase::add, py::arg("descriptors"))
        .def("query", &PyOrbDatabase::query, py::arg("descriptors"), py::arg("top_k") = 1)
        .def("save_database", &PyOrbDatabase::save_database, py::arg("path"));

    py::class_<PySiftDatabase>(m, "SiftDatabase")
        .def(py::init<int, int>(), py::arg("k") = 10, py::arg("L") = 6)
        .def("load_vocabulary", &PySiftDatabase::load_vocabulary, py::arg("path"))
        .def("create_vocabulary", &PySiftDatabase::create_vocabulary, py::arg("descriptors_list"))
        .def("add", &PySiftDatabase::add, py::arg("descriptors"))
        .def("query", &PySiftDatabase::query, py::arg("descriptors"), py::arg("top_k") = 1)
        .def("save_database", &PySiftDatabase::save_database, py::arg("path"));

    py::class_<PySuperpointDatabase>(m, "SuperpointDatabase")
        .def(py::init<int, int>(), py::arg("k") = 10, py::arg("L") = 6)
        .def("load_vocabulary", &PySuperpointDatabase::load_vocabulary, py::arg("path"))
        .def("create_vocabulary", &PySuperpointDatabase::create_vocabulary, py::arg("descriptors_list"))
        .def("add", &PySuperpointDatabase::add, py::arg("descriptors"))
        .def("query", &PySuperpointDatabase::query, py::arg("descriptors"), py::arg("top_k") = 1)
        .def("save_database", &PySuperpointDatabase::save_database, py::arg("path"));
}
