cmake_minimum_required(VERSION 3.21)
project(infer_depth_tutorial LANGUAGES CXX)

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

add_library(depth_utils STATIC
    utils/preprocess/preprocessor.cc
    utils/postprocess/depth_postprocess.cc
)
target_include_directories(depth_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/preprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/postprocess
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(depth_utils PUBLIC ${OpenCV_LIBS})

add_executable(infer-depth infer_depth.cc)
target_link_libraries(infer-depth PRIVATE depth_utils qbruntime)
