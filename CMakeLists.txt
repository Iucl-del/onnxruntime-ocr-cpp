cmake_minimum_required(VERSION 3.16)
project(OrtInference)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime)

add_executable(OrtInference main.cpp
        PaddleOCR.cpp
        PaddleOCR.hpp)
target_include_directories(OrtInference PRIVATE ${ONNXRUNTIME_DIR}/include)
target_link_directories(OrtInference PRIVATE ${ONNXRUNTIME_DIR}/lib)
target_link_libraries(OrtInference ${OpenCV_LIBS} onnxruntime)
