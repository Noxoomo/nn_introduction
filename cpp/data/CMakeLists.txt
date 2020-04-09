cmake_version()

project(trees LANGUAGES CXX)

add_library(data
        dataset.h
        binarized_dataset.h
        binarized_dataset.cpp
        grid.h
        grid.cpp
        grid_builder.cpp
        grid_builder.h
        histogram.h
        histogram.cpp
        load_data.h
        load_data.cpp
        )


enable_cxx17(data)
#maybe_enable_cuda(data)

target_link_libraries(data "${TORCH_LIBRARIES}" core util)


target_include_directories(data INTERFACE
        PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/
        )


install(TARGETS data EXPORT DataConfig
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


add_subdirectory(ut)
