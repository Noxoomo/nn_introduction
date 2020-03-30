cmake_version()
project(core_ut)

add_executable(core_ut context_ut.cpp matrix_ut.cpp multi_dim_arr_ut.cpp)
target_link_libraries(core_ut core vec_tools mx_tools gtest_main gtest)
add_test(core_ut core_ut COMMAND core_ut)
