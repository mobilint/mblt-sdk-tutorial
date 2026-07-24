cmake_minimum_required(VERSION 3.21)
project(infer_pose_tutorial LANGUAGES CXX)

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

add_library(pose_utils STATIC
    utils/preprocess/preprocessor.cc
    utils/postprocess/decode.cc
)
target_include_directories(pose_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/preprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/postprocess
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(pose_utils
    PUBLIC ${OpenCV_LIBS}
    PUBLIC Threads::Threads
)

# infer_pose.cc calls the qbruntime API directly, so link qbruntime here.
add_executable(infer-pose infer_pose.cc)
target_include_directories(infer-pose PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(infer-pose PRIVATE pose_utils qbruntime)
