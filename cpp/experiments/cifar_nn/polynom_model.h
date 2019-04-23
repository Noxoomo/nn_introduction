#pragma once

#include "model.h"
#include <models/polynom/polynom_autograd.h>

class PolynomModel : public experiments::Model {
public:

    PolynomModel(PolynomPtr polynom)
        : polynom_(polynom) {}

    PolynomModel() = default;

    torch::Tensor forward(torch::Tensor x) override;

    PolynomPtr polynom_;
};
