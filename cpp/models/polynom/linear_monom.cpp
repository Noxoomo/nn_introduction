#include "linear_monom.h"

#include "polynom.h"

LinearMonom::LinearMonom(PolynomStructure structure, const std::vector<double> &values, int origFId)
        : Monom(std::move(structure), values, origFId) {
}

Monom::MonomType LinearMonom::getMonomType() const {
    return Monom::MonomType::LinearMonom;
}

void LinearMonom::Forward(double lambda, ConstVecRef<float> features, VecRef<float> dst) const {
    // TODO lambda unused?

    for (const auto& split : Structure_.Splits) {
        if (!(features[split.Feature] > split.Condition)) {
            return;
        }
    }

    // TODO I don't understand why I need this `/ 2`
    for (int dim = 0; dim < (int)dst.size(); ++dim) {
        dst[dim] += features[origFId_] * Values_[dim] / 2;
    }
}

void LinearMonom::Backward(double lambda, ConstVecRef<float> features, ConstVecRef<float> outputsDer,
                           VecRef<float> featuresDer) const {
    // TODO lambda unused?

    // TODO we treat f = 0 as bias
    if (origFId_ == 0) {
        return;
    }

    for (const auto& split : Structure_.Splits) {
        if (!(features[split.Feature] > split.Condition)) {
            return;
        }
    }

    for (size_t dim = 0; dim < Values_.size(); ++dim) {
        featuresDer[origFId_] += Values_[dim] * outputsDer[dim];
    }
}

Polynom LinearTreesToPolynom(const Ensemble& ensemble) {
    PolynomBuilder builder;

    ensemble.visitModels([&](ModelPtr model) {
        auto lModel = std::dynamic_pointer_cast<LinearObliviousTree>(model);
        auto symmetricTrees = lModel->toSymmetricTrees();
        for (const auto& treePair : symmetricTrees) {
            builder.AddTree(std::get<0>(treePair), std::get<1>(treePair));
        }
    });

    return Polynom(Monom::MonomType::LinearMonom, builder.Build());
}
