#pragma once

#include "polynom.h"
#include <torch/torch.h>


using PolynomPtr = std::shared_ptr<Polynom>;

struct PolynomCuda {
    PolynomPtr Polynom_;

    torch::Tensor Features;
    torch::Tensor Conditions;
    torch::Tensor PolynomOffsets;
    torch::Tensor PolynomValues;
    torch::Tensor LeafSum;

    PolynomCuda(PolynomPtr polynom_);

    torch::Tensor Forward(torch::Tensor batch) const;
    torch::Tensor Backward(torch::Tensor batch, torch::Tensor backgrads) const;

};
