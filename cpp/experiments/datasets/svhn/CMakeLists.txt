cmake_version()
project(svhn)

include_directories(../../core)
include_directories(../../core/networks)

add_library(svhn
        svhn_reader.cpp
        svhn_reader.h)

target_link_libraries(svhn experiments_core)
