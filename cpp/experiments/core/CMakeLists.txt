cmake_version()
project(experiments_core)

find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(/contrib/json)

add_library(experiments_core
        em_like_train.h
        model.h
        model.cpp
        tensor_pair_dataset.h
        tensor_pair_dataset.cpp
        loss.cpp
        loss.h
        optimizer.cpp
        optimizer.h
        initializer.cpp
        initializer.h
#        linear_train.cpp
#        linear_train.h
        linear_model.cpp
        linear_model.h
        linear_function.cpp
        linear_function.h
        cross_entropy_loss.cpp
        cross_entropy_loss.h
        simple_conv_net.cpp
        simple_conv_net.h
        bce_sigm.cpp
        bce_sigm.h
        oblivious_tree_function.cpp
        oblivious_tree_function.h
        oblivious_tree_model.cpp
        oblivious_tree_model.h
#        oblivious_tree_train.cpp
#        oblivious_tree_train.h
        tree_train.h
        transform.cpp
        transform.h
        polynom_model.h
        polynom_model.cpp
        layer_norm.h
        layer_norm.cpp
        params.h

        networks/lenet.h
        networks/lenet.cpp
        networks/vgg.h
        networks/vgg.cpp
        networks/resnet.h
        networks/resnet.cpp
        networks/mobile_net_v2.h
        networks/mobile_net_v2.cpp
        networks/small_net.cpp
        networks/small_net.h
)
target_link_libraries(experiments_core "${TORCH_LIBRARIES}"  Eigen3::Eigen methods core models polynom targets util)
enable_cxx14(experiments_core)

# TODO move py module from here
#pybind11_add_module(experiments_core py_module.cpp)

#target_link_libraries(cifar_nn_py PRIVATE experiments_core)
#enable_cxx14(cifar_nn_py)
