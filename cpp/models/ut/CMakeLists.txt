cmake_version()
project(models_ut)

add_executable(models_ut models_ut.cpp)
target_link_libraries(models_ut core data vec_tools mx_tools trans funcs models polynom gtest_main)

add_test(models_ut models_ut COMMAND models_ut)
