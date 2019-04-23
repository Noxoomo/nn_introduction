#pragma once

#include "polynom.h"
#include <torch/torch.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/edge.h>

#include <cassert>
#include <iostream>
#include <util/parallel_executor.h>

using PolynomPtr = std::shared_ptr<Polynom>;

class PolynomBackward : public torch::autograd::Function {
public:
    PolynomBackward(torch::Tensor samplesBatch,
                    PolynomPtr polynom,
                    torch::autograd::edge_list&& nextEdges)
            : torch::autograd::Function(std::move(nextEdges))
            , samplesBatch_(std::move(samplesBatch))
            , polynom_(polynom) {

    }

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override;

private:
    torch::Tensor samplesBatch_;
    PolynomPtr polynom_;
};

class PolynomForward : public torch::autograd::Function {
public:

    PolynomForward(PolynomPtr polynom)
        : polynom_(std::move(polynom)){

    }

    torch::autograd::variable_list apply(torch::autograd::variable_list&& inputs) override;
private:

    PolynomPtr polynom_;
};


