cmake_version()

project(methods LANGUAGES CXX)

add_library(methods
        boosting.h
        boosting.cpp
        greedy_oblivious_tree.h
        greedy_oblivious_tree.cpp
        optimizer.h
        listener.h
        boosting_weak_target_factory.h
        boosting_weak_target_factory.cpp
        greedy_linear_oblivious_trees.h
        greedy_linear_oblivious_trees.cpp
)

#add_library(trees_kernels
#        kernels/empty.h
#        kernels/empty.cpp)


#maybe_enable_cuda(trees_kernels)

enable_cxx17(methods)
#maybe_enable_cuda(trees)

find_package(Eigen3 REQUIRED NO_MODULE)

target_link_libraries(methods "${TORCH_LIBRARIES}" core util models targets data Eigen3::Eigen)
#target_link_libraries(trees_kernels util)


target_include_directories(methods INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/
        )
#
#target_include_directories(trees_kernels INTERFACE
#        PUBLIC $<INSTALL_INTERFACE:include>
#        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels>
#        )

install(TARGETS methods EXPORT MethodsConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#install(TARGETS trees_kernels EXPORT TreesKernelsConfig
#        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#install(EXPORT CoreConfig DESTINATION share/CoreConfig/cmake)
#install(EXPORT CoreKernelsConfig DESTINATION share/CoreKernelsConfig/cmake)
#export(TARGETS core FILE CoreConfig.cmake)
#export(TARGETS core_kernels FILE CoreKernelsConfig.cmake)

add_subdirectory(ut)
