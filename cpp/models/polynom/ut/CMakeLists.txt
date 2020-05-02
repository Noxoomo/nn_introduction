cmake_version()
project(polynom_ut)

add_executable(polynom_ut polynom_ut.cpp)
target_link_libraries(polynom_ut core data vec_tools mx_tools trans funcs models methods polynom gtest_main)

add_test(polynom_ut polynom_ut COMMAND polynom_ut)
