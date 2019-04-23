cmake_version()
project(cifar_nn)

find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN_INCLUDE_DIR})

add_library(cifar_nn
        em_like_train.h
        cifar10_reader.cpp
        cifar10_reader.h
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
        linear_train.cpp
        linear_train.h
        linear_model.cpp
        linear_model.h
        linear_function.cpp
        linear_function.h
        cross_entropy_loss.cpp
        cross_entropy_loss.h
        lenet.cpp
        lenet.h
        vgg.cpp
        vgg.h
        resnet.cpp
        resnet.h
        simple_conv_net.cpp
        simple_conv_net.h
        bce_sigm.cpp
        bce_sigm.h
        oblivious_tree_function.cpp
        oblivious_tree_function.h
        oblivious_tree_model.cpp
        oblivious_tree_model.h
        oblivious_tree_train.cpp
        oblivious_tree_train.h
        tree_train.h
        transform.cpp
        transform.h
        polynom_model.h
        polynom_model.cpp
        )
target_link_libraries(cifar_nn "${TORCH_LIBRARIES}"  Eigen3::Eigen methods core models targets)
enable_cxx14(cifar_nn)

pybind11_add_module(cifar_nn_py py_module.cpp)
target_link_libraries(cifar_nn_py PRIVATE cifar_nn)
enable_cxx14(cifar_nn_py)

install(TARGETS cifar_nn EXPORT CifarNnConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
