cmake_version()

project(polynom LANGUAGES CXX)

add_library(polynom
        polynom.h
        polynom.cpp
        polynom_autograd.cpp
        polynom_autograd.h
)



#maybe_enable_cuda(models_kernels)

enable_cxx17(polynom)
maybe_enable_cuda(polynom)

target_link_libraries(polynom util)
#target_link_libraries(polynom_kernels util)


target_include_directories(polynom INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/
        )

#target_include_directories(polynom_kernels INTERFACE
#        PUBLIC $<INSTALL_INTERFACE:include>
#        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels>
#        )

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

