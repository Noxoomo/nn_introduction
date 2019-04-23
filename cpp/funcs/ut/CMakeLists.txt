cmake_version()
project(funcs_ut)

add_executable(funcs_ut func_ut.cpp soft_tree_test.cpp)
add_executable(soft_tree_ut soft_tree_test.cpp)
target_link_libraries(funcs_ut core vec_tools mx_tools trans  funcs gtest_main)
target_link_libraries(soft_tree_ut core vec_tools mx_tools trans funcs models gtest_main)

add_test(funcs_ut funcs_ut COMMAND funcs_ut)
add_test(soft_tree_ut soft_tree_ut COMMAND soft_tree_ut)
