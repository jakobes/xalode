cmake_minimum_required(VERSION 2.8.12)
project(xalode)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RELEASE)

# Boost
find_package(Boost REQUIRED)

# Armadillo
# find_package(Armadillo REQUIRED)

# pybin11
add_subdirectory(pybind11)

set(SOURCE_DIR "include")
include_directories(${SOURCE_DIR})

set(PYBIND11_PYTHON_VERSION 3.6)

set(CMAKE_CXX_FLAGS "-fPIC -shared ${CMAKE_CXX_FLAGS} -O3") # -std=c++17
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
pybind11_add_module(xalode SHARED src/main.cpp)

# Install headers for use with fenics
# add_library(xalode INTERFACE)
# target_include_directories(xalode INTERFACE ${SOURCE_DIR})

set(install_headers
    ${SOURCE_DIR}/cressman.h
    ${SOURCE_DIR}/forward_euler.h
    ${SOURCE_DIR}/utils.h
    )

set(include_dest "include/xalode")
install(FILES ${install_headers} DESTINATION ${include_dest})
