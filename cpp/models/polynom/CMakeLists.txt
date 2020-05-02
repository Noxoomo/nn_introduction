cmake_version()

project(polynom LANGUAGES CXX)

add_library(polynom
        monom.h
        monom.cpp
        sigmoid_prob_monom.h
        sigmoid_prob_monom.cpp
        exp_prob_monom.h
        exp_prob_monom.cpp
        linear_monom.h
        linear_monom.cpp
        polynom.h
        polynom.cpp
        polynom_autograd.cpp
        polynom_autograd.h
        polynom_gpu.h
        polynom_gpu.cpp
        soft_polynom.h
)



#maybe_enable_cuda(models_kernels)

enable_cxx17(polynom)



if (LINK_CUDA_POLYNOM)
    find_package(CUDA 10.0 REQUIRED)
    find_library(soft_polynom soft_polynom PATHS /usr/local/lib)
    if (NOT soft_polynom)
        message(FATAL_ERROR "soft_polynom is not found")
    else()
        message("found soft_polynom: ${soft_polynom}")
    endif()
    target_link_libraries(polynom util core soft_polynom ${CUDA_LIBRARIES})
else()
    add_library(polynom_cuda_stub no_soft_polynom.cpp)
    target_link_libraries(polynom util core polynom_cuda_stub)
endif()



target_include_directories(polynom INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/
        )


install(TARGETS polynom EXPORT PolynomConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#install(TARGETS models_kernels EXPORT TreesKernelsConfig
#        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

#install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#install(EXPORT CoreConfig DESTINATION share/CoreConfig/cmake)
#install(EXPORT CoreKernelsConfig DESTINATION share/CoreKernelsConfig/cmake)
#export(TARGETS core FILE CoreConfig.cmake)
#export(TARGETS core_kernels FILE CoreKernelsConfig.cmake)

add_subdirectory(ut)
