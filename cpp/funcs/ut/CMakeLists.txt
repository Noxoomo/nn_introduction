cmake_version()
project(funcs_ut)

add_executable(funcs_ut func_ut.cpp)
target_link_libraries(funcs_ut core vec_tools mx_tools trans  funcs gtest_main)

add_test(funcs_ut funcs_ut COMMAND funcs_ut)
