cmake_minimum_required(VERSION 3.18)
project(infer_det_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_FLAGS "-O3 -march=armv8-a -ffast-math")
set(CMAKE_CXX_FLAGS "-O3 -march=armv8-a -ffast-math")

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_library(det_utils STATIC
    utils/inference/parser.cc
    utils/inference/runner.cc
    utils/inference/transform.cc
    utils/postprocess/decode.cc
)
target_include_directories(det_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/postprocess
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(det_utils
    PUBLIC ${OpenCV_LIBS}
    PUBLIC Threads::Threads
    PRIVATE qbruntime
)

add_executable(infer-det infer_det.cc)
target_include_directories(infer-det PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(infer-det PRIVATE det_utils)
