cmake_version()

project(targets LANGUAGES CXX)

add_library(targets
        target.h
        target.cpp
        l2.h
        l2.cpp
        stat_based_loss.h

        cross_entropy.cpp
        cross_entropy.h

        additive_statistics.h

        linear_l2.h
        linear_l2.cpp
        linear_l2_stat.h
        linear_l2_stat.cpp
)



#maybe_enable_cuda(trees_kernels)

enable_cxx17(targets)
#maybe_enable_cuda(trees)

find_package(Eigen3 REQUIRED NO_MODULE)

target_link_libraries(targets "${TORCH_LIBRARIES}" core util data vec_tools trans funcs Eigen3::Eigen)
#target_link_libraries(trees_kernels util)


target_include_directories(targets INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/
        )
#
#target_include_directories(trees_kernels INTERFACE
#        PUBLIC $<INSTALL_INTERFACE:include>
#        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels>
#        )

install(TARGETS targets EXPORT TargetsConfig
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
