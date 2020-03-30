cmake_version()
project(cifar_networks)

include_directories(../../experiments)

# train

add_executable(train_default train_default.cpp common.h common.cpp)
target_link_libraries(train_default "${TORCH_LIBRARIES}" experiments)

add_executable(train_default_em train_default_em.cpp common.h common.cpp common_em.h)
target_link_libraries(train_default_em "${TORCH_LIBRARIES}" experiments)

add_executable(train_with_catboost train_with_catboost.cpp common.h common.cpp catboost_nn.h catboost_nn.cpp)
target_link_libraries(train_with_catboost "${TORCH_LIBRARIES}" experiments polynom ${CATBOOST})

add_executable(train_cnn_linear train_cnn_linear_trees.cpp common.h common.cpp)
target_link_libraries(train_cnn_linear "${TORCH_LIBRARIES}" experiments methods)

# misc

add_executable(check_cuda_libtorch check_cuda_libtorch.cpp)
target_link_libraries(check_cuda_libtorch "${TORCH_LIBRARIES}")

add_executable(test_linear_trees_app test_linear_trees.cpp)
target_link_libraries(test_linear_trees_app "${TORCH_LIBRARIES}" core methods data ${CATBOOST})
