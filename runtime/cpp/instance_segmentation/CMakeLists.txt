cmake_minimum_required(VERSION 3.21)
project(infer_seg_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_C_FLAGS   "-O3 -march=armv8-a -ffast-math")
    set(CMAKE_CXX_FLAGS "-O3 -march=armv8-a -ffast-math")
else()
    set(CMAKE_C_FLAGS   "-O3 -march=native -ffast-math")
    set(CMAKE_CXX_FLAGS "-O3 -march=native -ffast-math")
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_library(seg_utils STATIC
    utils/inference/runner.cc
    utils/inference/transform.cc
    utils/postprocess/decode.cc
)
target_include_directories(seg_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/inference
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/postprocess
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(seg_utils
    PUBLIC ${OpenCV_LIBS}
    PUBLIC Threads::Threads
    PRIVATE qbruntime
)

add_executable(infer-seg infer_seg.cc)
target_include_directories(infer-seg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(infer-seg PRIVATE seg_utils)
