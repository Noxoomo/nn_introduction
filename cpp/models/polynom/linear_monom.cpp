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
        if (features[split.Feature] <= split.Condition) {
            return;
        }
    }

    for (int dim = 0; dim < (int)dst.size(); ++dim) {
        if (origFId_ != -1) {
            dst[dim] += features[origFId_] * Values_[dim];
        } else {
            dst[dim] += Values_[dim];
        }
    }
}

void LinearMonom::Backward(double lambda, ConstVecRef<float> features, ConstVecRef<float> outputsDer,
                           VecRef<float> featuresDer) const {
    // TODO lambda unused?

    // TODO we treat f = 0 as bias
    if (origFId_ == -1) {
        return;
    }

    for (const auto& split : Structure_.Splits) {
        if (features[split.Feature] <= split.Condition) {
            return;
        }
    }

    for (size_t dim = 0; dim < Values_.size(); ++dim) {
        featuresDer[origFId_] += Values_[dim] * outputsDer[dim];
    }
}

std::vector<std::tuple<TSymmetricTree, int>> LinearToSymmetricTrees(const LinearObliviousTree& loTree) {
    std::vector<std::tuple<TSymmetricTree, int>> res;

    int i = 0;
    for (int origFId : loTree.leaves_[0].usedFeaturesInOrder_) {
        TSymmetricTree tree;
        for (const auto& [splitOrigFId, splitCond] : loTree.splits_) {
            tree.Features.push_back(splitOrigFId);
            tree.Conditions.push_back(splitCond);
        }

        for (const auto& l : loTree.leaves_) {
            tree.Leaves.push_back(l.w_(i, 0) * loTree.scale_);
            tree.Weights.push_back(0); // TODO do we need to keep weights?
        }

        res.emplace_back(std::make_tuple(std::move(tree), origFId));
        ++i;
    }

    return res;
}

Polynom LinearTreesToPolynom(const Ensemble& ensemble) {
    PolynomBuilder builder;

    ensemble.visitModels([&](ModelPtr model) {
        auto lModel = std::dynamic_pointer_cast<LinearObliviousTree>(model);
        auto symmetricTrees = LinearToSymmetricTrees(*lModel);
        for (const auto& treePair : symmetricTrees) {
            builder.AddTree(std::get<0>(treePair), std::get<1>(treePair));
        }
    });

    return Polynom(Monom::MonomType::LinearMonom, builder.Build());
}
