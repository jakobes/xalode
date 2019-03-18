cmake_minimum_required(VERSION 2.8.12)
project(bbidomain)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RELEASE)

# Boost and its components
# find_package(Boost REQUIRED)
# include_directories(${BOOST_USE_FILES})

# pybin11
add_subdirectory(external/pybind11)

set(SOURCE_DIR "include")
include_directories(${SOURCE_DIR})

set(PYBIND11_PYTHON_VERSION 3.6)

set(CMAKE_CXX_FLAGS "-fPIC -shared ${CMAKE_CXX_FLAGS} -O3") # -std=c++17
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
pybind11_add_module(bbidomain SHARED src/main.cpp)
