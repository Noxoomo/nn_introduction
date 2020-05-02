cmake_version()

project(monoms LANGUAGES CXX)

add_library(monoms
        monom.h
        sigmoid_prob_monom.h
        sigmoid_prob_monom.cpp
        exp_prob_monom.h
        exp_prob_monom.cpp
        
        )

target_link_libraries(polynom monoms)

#maybe_enable_cuda(models_kernels)

enable_cxx17(monoms)
