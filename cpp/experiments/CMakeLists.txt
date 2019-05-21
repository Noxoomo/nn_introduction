cmake_version()
project(experiments)

add_subdirectory(core)
add_subdirectory(datasets)

add_library(experiments dummy.cpp)
target_link_libraries(experiments experiments_core cifar10 mnist svhn)

install(TARGETS experiments EXPORT ExperimentsConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
