#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <memory>
#include <opencv2/core.hpp>
#include <stdexcept>

#include "DBoW2.h"

namespace py = pybind11;

// design a function to convert numpy arry to vector of cv::Mat
// ORB descriptors are 32 bytes
static std::vector<cv::Mat> numpy_to_features(
    py::array_t<unsigned char, py::array::c_style | py::array::forcecast> descriptors) {
    auto buf = descriptors.request();
    
    // ORB descriptors should have 32 columns, each unit of the descriptor is 8 bits, 
    // total 256 bits per descriptor.
    // The OpenCV ORB descriptors shape is (N, 32), N is keypoints number in the image. 
    // Each row corresponds to a descriptor for a keypoint
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

// SIFT descriptors: (N, 128) float32 → std::vector<std::vector<float>>
static std::vector<std::vector<float>> numpy_to_sift_features(
    py::array_t<float, py::array::c_style | py::array::forcecast> descriptors) {
    auto buf = descriptors.request();
    if (buf.ndim != 2 || buf.shape[1] != 128)
        throw std::runtime_error("Input should be a 2D array with shape (N, 128)");

    auto *ptr = static_cast<float *>(buf.ptr);
    int rows = static_cast<int>(buf.shape[0]);

    std::vector<std::vector<float>> features;
    features.reserve(rows);
    for (int i = 0; i < rows; ++i)
        features.emplace_back(ptr + i * 128, ptr + (i + 1) * 128);
    return features;
}

// ORB database
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

        void load_vocabulary(const std::string &path) {
            OrbVocabulary voc;
            if(path.size() >= 4 && path.substr(path.size() - 4) == ".txt")
                voc.loadFromTextFile(path);
            else
                voc.load(path);
            vocabulary_ = voc;
            database_ = std::make_unique<OrbDatabase>(vocabulary_, false, 0);
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

// SIFT database
class PySiftDatabase {
public:
    PySiftDatabase(int k = 10, int L = 6)
        : vocabulary_(k, L, DBoW2::TF_IDF, DBoW2::L2_NORM), database_(nullptr) {}

    void load_vocabulary(const std::string &path) {
        SiftVocabulary voc;
        if(path.size() >= 4 && path.substr(path.size() - 4) == ".txt")
            voc.loadFromTextFile(path);
        else
            voc.load(path);
        vocabulary_ = voc;
        database_ = std::make_unique<SiftDatabase>(vocabulary_, false, 0);
    }

    void create_vocabulary(
        const std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> &descriptors_list) {
        std::vector<std::vector<std::vector<float>>> all_features;
        all_features.reserve(descriptors_list.size());
        for (const auto &d : descriptors_list)
            all_features.push_back(numpy_to_sift_features(d));
        vocabulary_.create(all_features);
        database_ = std::make_unique<SiftDatabase>(vocabulary_, false, 0);
    }

    int add(py::array_t<float, py::array::c_style | py::array::forcecast> descriptors) {
        if (!database_) throw std::runtime_error("Vocabulary must be created before adding features to the database.");
        return database_->add(numpy_to_sift_features(descriptors));
    }

    std::vector<std::pair<int, double>> query(
        py::array_t<float, py::array::c_style | py::array::forcecast> descriptors, int top_k = 1) {
        if (!database_) throw std::runtime_error("Vocabulary must be created before querying the database.");
        DBoW2::QueryResults results;
        database_->query(numpy_to_sift_features(descriptors), results, top_k);
        std::vector<std::pair<int, double>> output;
        for (const auto &r : results) output.emplace_back(r.Id, r.Score);
        return output;
    }

    void save(const std::string &path) {
        if (!database_) throw std::runtime_error("Vocabulary must be created before saving the database.");
        database_->save(path);
    }

private:
    SiftVocabulary vocabulary_;
    std::unique_ptr<SiftDatabase> database_;
};

PYBIND11_MODULE(dbow2_cpp, m) {
    py::class_<PyOrbDatabase>(m, "OrbDatabase")
        .def(py::init<int, int>(), py::arg("k") = 9, py::arg("L") = 3)
        .def("load_vocabulary", &PyOrbDatabase::load_vocabulary, py::arg("path"))
        .def("create_vocabulary", &PyOrbDatabase::create_vocabulary, py::arg("descriptors_list"))
        .def("add", &PyOrbDatabase::add, py::arg("descriptors"))
        .def("query", &PyOrbDatabase::query, py::arg("descriptors"), py::arg("top_k") = 1)
        .def("save", &PyOrbDatabase::save, py::arg("path"));

    py::class_<PySiftDatabase>(m, "SiftDatabase")
        .def(py::init<int, int>(), py::arg("k") = 10, py::arg("L") = 6)
        .def("load_vocabulary", &PySiftDatabase::load_vocabulary, py::arg("path"))
        .def("create_vocabulary", &PySiftDatabase::create_vocabulary, py::arg("descriptors_list"))
        .def("add", &PySiftDatabase::add, py::arg("descriptors"))
        .def("query", &PySiftDatabase::query, py::arg("descriptors"), py::arg("top_k") = 1)
        .def("save", &PySiftDatabase::save, py::arg("path"));
}
