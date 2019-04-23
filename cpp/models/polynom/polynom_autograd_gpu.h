#pragma once

#include <torch/torch.h>
#include <torch/csrc/autograd/function.h>
#include <models/polynom/soft_polynom.cuh>
#include <models/polynom/polynom.h>

using PolynomPtr = std::shared_ptr<Polynom>;

class PolynomBackwardGPU : public torch::autograd::Function {
    torch::autograd::variable_list apply(torch::autograd::variable_list &&inputs) override {
        // todo
    };
}
class PolynomForwardGPU : public torch::autograd::Function {
private:
    int polynomCount;
    int outputDim;
    torch::Tensor gpuSplits;
    PolynomPtr polynom_;

public:
    PolynomForwardGPU(const PolynomPtr& polynom) {
        std::vector<int> splits;
        std::vector<float> conditions;
        std::vector<int> polynomDepth;
        std::vector<int> polynomOffsets;
        std::vector<float> values;
        polynomCount = polynom->Ensemble_.size();
        outputDim = polynom->OutDim();
        polynom_ = polynom;
        for (const Monom& monom: polynom->Ensemble_) {
            polynomOffsets.push_back(splits.size());
            polynomDepth.push_back(monom.Structure_.GetDepth());
            for (const BinarySplit& binarySplit: monom.Structure_.Splits) {
                splits.push_back(binarySplit.Feature);
                conditions.push_back(binarySplit.Condition);
            }
            values.insert(values.end(), monom.Values_.begin(), monom.Values_.end());
        }
    }

    torch::autograd::variable_list apply(torch::autograd::variable_list &&inputs) override {
        return PolynomForward()

    }
};
