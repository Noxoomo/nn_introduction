#pragma once

#include "catboost_api.h"
#include <vector>
#include <string>
#include <cstddef>

struct TSymmetricTree {
    std::vector<int> Features;
    std::vector<float> Conditions;
    std::vector<float> Leaves;
    std::vector<float> Weights;

    int OutputDim() const {
        return Weights.size() ? Leaves.size() / Weights.size() : 0;
    }

    void Forward(const float* features, float* dst) const {
        int bin = 0;
        for (int k = 0; k < Conditions.size(); ++k) {
            if (features[Features[k]] > Conditions[k]) {
                bin |= 1 << k;
            }
        }
        for (int dim = 0; dim < OutputDim(); ++dim) {
            dst[dim] += Leaves[bin * OutputDim() + dim];
        }

    }
};

struct TEnsemble {
    std::vector<TSymmetricTree> Trees;
    void Forward(const float* features, float* dst) {
        for (const auto& tree : Trees) {
            tree.Forward(features, dst);
        }
    }
};



struct TPool {
    const float* Features = nullptr;
    const float* Labels = nullptr;
    const float* Weights = nullptr;
    size_t FeaturesCount = 0;
    size_t SamplesCount = 0;
};



inline TEnsemble Train(const TPool& learn,
                        const TPool& test,
                        const std::string& paramsJson) {
    ResultHandle handle;
    TrainCatBoost(learn.Features,
                  learn.Labels,
                  learn.Weights,
                  learn.FeaturesCount,
                  learn.SamplesCount,
                  test.Features, test.Labels,
                  test.Weights,
                  test.SamplesCount,
                  paramsJson.data(),
                  &handle
        );

    if (!handle) {
        throw std::exception();
    }

    TEnsemble ensemble;
    ensemble.Trees.resize(TreesCount(handle));
    int outputDim = OutputDim(handle);
    int numTrees = ensemble.Trees.size();
    for (int tree = 0; tree < numTrees; ++tree) {
        auto depth = TreeDepth(handle, tree);

        auto& currentTree = ensemble.Trees[tree];
        currentTree.Features.resize(depth);
        currentTree.Conditions.resize(depth);
        currentTree.Leaves.resize(outputDim * (1 << depth));
        currentTree.Weights.resize((1 << depth));

        CopyTree(handle,
            tree,
            currentTree.Features.data(),
            currentTree.Conditions.data(),
            currentTree.Leaves.data(),
            currentTree.Weights.data()
            );
    }


    FreeHandle(&handle);
    return ensemble;
}
