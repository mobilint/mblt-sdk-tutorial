cmake_minimum_required(VERSION 3.18)
project(infer_cls_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_FLAGS "-O3 -march=armv8-a -ffast-math")
set(CMAKE_CXX_FLAGS "-O3 -march=armv8-a -ffast-math")

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(infer-cls infer_cls.cc)
target_link_libraries(infer-cls PRIVATE ${OpenCV_LIBS} Threads::Threads qbruntime)
