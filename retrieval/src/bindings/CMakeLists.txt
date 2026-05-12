cmake_minimum_required(VERSION 3.14)
project(dbow2_python)

find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

set(BUILD_Demo OFF CACHE BOOL "Do not build the DBoW2 demo for Python bindings" FORCE)

add_subdirectory(
    "${REPO_ROOT}/external_repositories/DBoW2"
    "${CMAKE_CURRENT_BINARY_DIR}/DBoW2-build"
)

pybind11_add_module(dbow2_cpp
    dbow2_pybind.cpp
)

set_target_properties(dbow2_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(dbow2_cpp PRIVATE
    "${REPO_ROOT}/external_repositories/DBoW2/include"
    "${REPO_ROOT}/external_repositories/DBoW2/include/DBoW2"
)

target_link_libraries(dbow2_cpp PRIVATE
    DBoW2
    ${OpenCV_LIBS}
)
