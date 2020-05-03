cmake_version()
project(train_linear_ot)

add_executable(train_linear_ot fit.cpp)
add_executable(apply_linear_ot apply.cpp)

#cmake_policy(SET CMP0069 NEW)
#include(CheckIPOSupported)
#check_ipo_supported()
#set_target_properties(train_ot PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

target_link_libraries(train_linear_ot methods "${TORCH_LIBRARIES}" core util models targets data)
target_link_libraries(apply_linear_ot methods "${TORCH_LIBRARIES}" core util models targets data)
