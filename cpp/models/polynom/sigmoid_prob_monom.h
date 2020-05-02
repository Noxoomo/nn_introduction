#pragma once

#include "monom.h"

// P(x) = sigma(x)
class SigmoidProbMonom : public Monom {
public:
    SigmoidProbMonom() = default;

    SigmoidProbMonom(PolynomStructure structure, std::vector<double> values)
            : Monom(std::move(structure), std::move(values)) {

    }

    Monom::MonomType getMonomType() const override;

    void Forward(double lambda, ConstVecRef<float> features, VecRef<float> dst) const override;
    void Backward(double lambda, ConstVecRef<float> features, ConstVecRef<float> outputsDer, VecRef<float> featuresDer) const override;
};
