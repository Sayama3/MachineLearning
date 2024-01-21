cmake_minimum_required(VERSION 3.24)

project(Eigen3 VERSION 3.4.0 LANGUAGES CXX)

file(GLOB_RECURSE EIGEN_SRC
    include/*
)

add_library(Eigen STATIC ${EIGEN_SRC})

target_include_directories(Eigen PUBLIC include)
add_library(Eigen3::Eigen ALIAS Eigen)

set_target_properties(Eigen PROPERTIES
        VERSION       3.4.0
        PUBLIC_HEADER "include/Eigen/Eigen"
        PUBLIC cxx_std_20
        LINKER_LANGUAGE CXX
        LANGUAGES CXX
)