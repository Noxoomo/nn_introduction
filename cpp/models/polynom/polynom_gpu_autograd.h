#pragma once

#include "polynom_gpu.h"
#include <torch/torch.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/edge.h>

#include <cassert>
#include <iostream>
#include <util/parallel_executor.h>

using PolynomCudaPtr = std::shared_ptr<PolynomCuda>;

class PolynomBackwardGPU : public torch::autograd::Function {
public:
    PolynomBackwardGPU(torch::Tensor samplesBatch,
                       PolynomCudaPtr polynom,
                    torch::autograd::edge_list&& nextEdges)
            : torch::autograd::Function(std::move(nextEdges))
            , samplesBatch_(std::move(samplesBatch))
            , polynom_(polynom) {

    }

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override;

private:
    torch::Tensor samplesBatch_;
    PolynomCudaPtr polynom_;
};

class PolynomForwardGPU : public torch::autograd::Function {
public:
    PolynomForwardGPU(PolynomCudaPtr polynom)
        : polynom_(std::move(polynom)){

    }

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override;
private:

    PolynomCudaPtr polynom_;
};


