#pragma once

#include "monom.h"

#include <catboost_wrapper.h>
#include <memory>
#include <unordered_map>
#include <util/array_ref.h>
#include <util/city.h>

// TODO everything about origFId is hack to make linear trees work. Need to rework all this
// but I don't wanna do it right now, I'm too lazy

struct TStat {
    std::vector<double> Value;
    double Weight = -1;
};

// sum v * Prod [x _i > c_i]
class PolynomBuilder {
public:
    void AddTree(const TSymmetricTree& tree, int origFId = -1);

    PolynomBuilder& AddEnsemble(const TEnsemble& ensemble) {
        for (const auto& tree : ensemble.Trees) {
            AddTree(tree);
        }
        return *this;
    }

    std::unordered_map<std::tuple<PolynomStructure, int>, TStat> Build() {
        return EnsemblePolynoms;
    }

private:
    std::unordered_map<std::tuple<PolynomStructure, int>, TStat> EnsemblePolynoms;
};

struct Polynom {
    std::vector<MonomPtr> Ensemble_;
    double Lambda_  = 1.0;

    Polynom(Monom::MonomType monomType, const std::unordered_map<std::tuple<PolynomStructure, int>, TStat>& polynom) {
        for (const auto& [structure, stat] : polynom) {
            Ensemble_.emplace_back(Monom::createMonom(monomType,
                    std::get<0>(structure),
                    stat.Value,
                    std::get<1>(structure)));
        }
    }

    Polynom() = default;

    void PrintHistogram();

    Monom::MonomType getMonomType() const;

    //forward/backward will append to dst
    void Forward(ConstVecRef<float> features, VecRef<float> dst) const;
    void Backward(ConstVecRef<float> features, ConstVecRef<float> outputsDer, VecRef<float> featuresDer) const;

    int OutDim() const {
        return Ensemble_.empty() ? 0 : Ensemble_.back()->OutDim();
    }
};

using PolynomPtr = std::shared_ptr<Polynom>;
