#pragma once

#include "model.h"
#include <models/polynom/polynom_gpu.h>
#include <models/polynom/polynom_autograd.h>
#include <models/polynom/monom.h>

class PolynomModel : public experiments::Model {
public:
    explicit PolynomModel(PolynomPtr polynom)
            : Model()
            , polynom_(std::move(polynom))
            , monomType_(polynom_->getMonomType()) {

    }

    PolynomModel(Monom::MonomType monomType)
            : Model()
            , monomType_(monomType) {

    }

    torch::Tensor forward(torch::Tensor x) override;

    void reset(PolynomPtr polynom) {
        polynom_ = std::move(polynom);
        polynomCuda_ = nullptr;
    }

    void setLambda(double lambda) {
        polynom_->Lambda_ = lambda;
    }

    Monom::MonomType getMonomType() {
        return monomType_;
    }

private:
    PolynomPtr polynom_;
    PolynomCudaPtr polynomCuda_;
    Monom::MonomType monomType_;

};
