#pragma once

#include <vector>

#include "monom.h"
#include "polynom.h"
#include <models/linear_oblivious_tree.h>
#include <models/ensemble.h>

class LinearMonom : public Monom {
public:
    LinearMonom() = default;

    LinearMonom(PolynomStructure structure, const std::vector<double>& values, int origFId);

    Monom::MonomType getMonomType() const override;

    void Forward(double lambda, ConstVecRef<float> features, VecRef<float> dst) const override;
    void Backward(double lambda, ConstVecRef<float> features, ConstVecRef<float> outputsDer, VecRef<float> featuresDer) const override;

};

std::vector<std::tuple<TSymmetricTree, int>> LinearToSymmetricTrees(const LinearObliviousTree& loTree, double scale);
Polynom LinearTreesToPolynom(const Ensemble& ensemble);
