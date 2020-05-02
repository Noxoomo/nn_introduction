#pragma once

#include "polynom.h"
#include <torch/torch.h>



struct PolynomCuda {
    PolynomPtr Polynom_;

    torch::Tensor Features;
    torch::Tensor Conditions;
    torch::Tensor PolynomOffsets;
    torch::Tensor PolynomValues;
    torch::Tensor OrigFIds;

    PolynomCuda(PolynomPtr polynom_);

    torch::Tensor Forward(torch::Tensor batch) const;

    torch::Tensor Backward(torch::Tensor features, torch::Tensor outputDer) const;
};

using PolynomCudaPtr = std::shared_ptr<PolynomCuda>;

