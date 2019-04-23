#pragma once

#include <torch/torch.h>
#include <torch/csrc/autograd/function.h>
#include <models/polynom/soft_polynom.cuh>
#include <models/polynom/polynom.h>

class SoftTreeCudaBackward : public torch::autograd::Function {
    torch::autograd::variable_list apply(torch::autograd::variable_list &&inputs) override {
        // todo
    };
}
class SoftTreeCuda : public torch::autograd::Function {
private:
    int polynomCount;
    int outputDim;
    torch::Tensor gpuSplits;

public:
    SoftTreeCuda(const Polynom& polynom) {
        std::vector<int> splits;
        std::vector<float> conditions;
        std::vector<int> polynomDepth;
        std::vector<int> polynomOffsets;
        std::vector<float> values;
        polynomCount = polynom.Ensemble_.size();
        outputDim = 2; // todo: fix
        for (const Monom& monom: polynom.Ensemble_) {
            polynomOffsets.push_back(splits.size());
            polynomDepth.push_back(monom.Structure_.GetDepth());
            for (const BinarySplit& binarySplit: monom.Structure_.Splits) {
                splits.push_back(binarySplit.Feature);
                conditions.push_back(binarySplit.Condition);
            }
            values.insert(values.end(), monom.Values_.begin(), monom.Values_.end());
        }

        gpuSplits = torch::from_blob(splits.data(), {1});



    }

    torch::autograd::variable_list apply(torch::autograd::variable_list &&inputs) override {
        torch::autograd::Variable x = inputs[0];
        auto sz = x.sizes();
        torch::autograd::Variable res = torch::zeros({sz[0]}, torch::kFloat32);
        int fCount = 6;
        float features[6] = {0,0,0,0,0,0};
        int batchSize = 1;
        int polynomCount = 1;
        float probs[1];
        float out[1];
        PolynomProbsImpl(
                features,
                fCount,
                batchSize,
                splits,
                conditions,
                polynomOffsets,
                polynomDepth,
                polynomCount,
                probs);
        PolynomForwardImpl(
                probs,
                batchSize,
                values,
                polynomCount,
                outputDim,
                out);
    }
};
