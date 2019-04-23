cmake_version()
project(cifar_networks)

include_directories(../../experiments)

add_executable(lenet lenet.cpp common.h common.cpp vgg_em.cpp common_em.h)
target_link_libraries(lenet "${TORCH_LIBRARIES}" cifar_nn)


add_executable(lenet_em lenet_em.cpp common.h common.cpp common_em.h)
target_link_libraries(lenet_em "${TORCH_LIBRARIES}" cifar_nn)

add_executable(lenet_save_restore lenet_save_restore.cpp)
target_link_libraries(lenet_save_restore "${TORCH_LIBRARIES}" cifar_nn)

add_executable(vgg vgg.cpp common.h common.cpp)
target_link_libraries(vgg "${TORCH_LIBRARIES}" cifar_nn)

add_executable(vgg_em vgg_em.cpp common.h common.cpp common_em.h)
target_link_libraries(vgg_em "${TORCH_LIBRARIES}" cifar_nn)

add_executable(vgg_save_restore vgg_save_restore.cpp)
target_link_libraries(vgg_save_restore "${TORCH_LIBRARIES}" cifar_nn)

add_executable(resnet resnet.cpp common.h common.cpp)
target_link_libraries(resnet "${TORCH_LIBRARIES}" cifar_nn)

add_executable(resnet_em resnet_em.cpp common.h common.cpp common_em.h)
target_link_libraries(resnet_em "${TORCH_LIBRARIES}" cifar_nn)

add_executable(resnet_save_restore resnet_save_restore.cpp)
target_link_libraries(resnet_save_restore "${TORCH_LIBRARIES}" cifar_nn)

add_executable(check_cuda_libtorch check_cuda_libtorch.cpp)
target_link_libraries(check_cuda_libtorch "${TORCH_LIBRARIES}")


add_executable(lenet_catboost lenet_catboost.cpp common.h common.cpp catboost_nn.h catboost_nn.cpp)
target_link_libraries(lenet_catboost "${TORCH_LIBRARIES}" cifar_nn  polynom ${CATBOOST})

add_executable(resnet_catboost resnet_catboost.cpp common.h common.cpp catboost_nn.h catboost_nn.cpp)
target_link_libraries(resnet_catboost "${TORCH_LIBRARIES}" cifar_nn  polynom ${CATBOOST})


add_executable(vgg_catboost vgg_catboost.cpp common.h common.cpp catboost_nn.h catboost_nn.cpp)
target_link_libraries(vgg_catboost "${TORCH_LIBRARIES}" cifar_nn  polynom ${CATBOOST})
