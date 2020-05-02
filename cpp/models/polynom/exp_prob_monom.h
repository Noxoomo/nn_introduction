#pragma once

#include "monom.h"

// P(x) = 1 - e^{-x}
class ExpProbMonom : public Monom {
public:
    ExpProbMonom() = default;

    ExpProbMonom(PolynomStructure structure, std::vector<double> values)
            : Monom(std::move(structure), std::move(values)) {

    }

    Monom::MonomType getMonomType() const override;

    void Forward(double lambda, ConstVecRef<float> features, VecRef<float> dst) const override;
    void Backward(double lambda, ConstVecRef<float> features, ConstVecRef<float> outputsDer, VecRef<float> featuresDer) const override;
};
