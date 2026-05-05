#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <memory>
#include <opencv2/core.hpp>
#include <stdexcept>

#include "DBoW2.h"

namespace py = pybind11;

static std::vector<cv::Mat> numpy_to_features(
    py::array_t<unsigned char, py::array::c_style | py::array::forcecast> descriptors) {
    auto buf = descriptors.request();

    if (buf.ndim != 2 || buf.shape[1] != 32) {
        throw std::runtime_error("Input should be a 2D array with shape (N, 32)");
    }

    auto *ptr = static_cast<unsigned char *>(buf.ptr);
    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);

    cv::Mat mat(rows, cols, CV_8U, ptr);

    std::vector<cv::Mat> features;
    features.reserve(rows);

    for (int i = 0; i < rows; ++i) {
        features.push_back(mat.row(i).clone());
    }

    return features;
}

class PyOrbDatabase {
    public:
        PyOrbDatabase(int k = 9, int L = 3) : vocabulary_(k, L, DBoW2::TF_IDF, DBoW2::L1_NORM), database_(nullptr) {}

        void create_vocabulary(
            const std::vector<py::array_t<unsigned char, py::array::c_style | py::array::forcecast>> &descriptors_list) {
            std::vector<std::vector<cv::Mat>> all_features;
            all_features.reserve(descriptors_list.size());

            for (const auto &descriptors : descriptors_list) {
                all_features.push_back(numpy_to_features(descriptors));
            }

            vocabulary_.create(all_features);
            database_ = std::make_unique<OrbDatabase>(vocabulary_, false, 0);
        }

        int add(py::array_t<unsigned char, py::array::c_style | py::array::forcecast> descriptors) {
            if (!database_) {
                throw std::runtime_error("Vocabulary must be created before adding features to the database.");
            }

            auto features = numpy_to_features(descriptors);
            return database_->add(features);
        }

        std::vector<std::pair<int, double>> query(
            py::array_t<unsigned char, py::array::c_style | py::array::forcecast> descriptors, int top_k = 1) {
            if (!database_) {
                throw std::runtime_error("Vocabulary must be created before querying the database.");
            }

            auto features = numpy_to_features(descriptors);

            DBoW2::QueryResults results;
            database_->query(features, results, top_k);

            std::vector<std::pair<int, double>> output;

            for (const auto &result : results) {
                output.emplace_back(result.Id, result.Score);
            }

            return output;
        }

        void save(const std::string &path) {
            if (!database_) {
                throw std::runtime_error("Vocabulary must be created before saving the database.");
            }

            database_->save(path);
        }
    private:
        OrbVocabulary vocabulary_;
        std::unique_ptr<OrbDatabase> database_;
};

PYBIND11_MODULE(dbow2_cpp, m) {
    py::class_<PyOrbDatabase>(m, "OrbDatabase")
        .def(py::init<int, int>(), py::arg("k") = 9, py::arg("L") = 3)
        .def("create_vocabulary", &PyOrbDatabase::create_vocabulary, py::arg("descriptors_list"))
        .def("add", &PyOrbDatabase::add, py::arg("descriptors"))
        .def("query", &PyOrbDatabase::query, py::arg("descriptors"), py::arg("top_k") = 1)
        .def("save", &PyOrbDatabase::save, py::arg("path"));
}
