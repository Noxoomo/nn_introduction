#include "polynom_gpu.h"
#include "soft_polynom.h"
#include <core/buffer.h>

PolynomCuda::PolynomCuda(PolynomPtr polynom)
: Polynom_(polynom) {

    std::sort(Polynom_->Ensemble_.begin(), Polynom_->Ensemble_.end(), [&](const Monom& left, const Monom& right) {
        return left.Structure_.GetDepth() < right.Structure_.GetDepth();
    });

    std::vector<int> flatFeatureIds;
    std::vector<float> conditions;
    std::vector<int>  offsets;
    std::vector<float> values;

    int cursor = 0;
    for (const auto& monom : Polynom_->Ensemble_) {
        for (const auto& split : monom.Structure_.Splits) {
            flatFeatureIds.push_back(split.Feature);
            conditions.push_back(split.Condition);
        }
        values.insert(values.end(), monom.Values_.begin(), monom.Values_.end());
        offsets.push_back(cursor);
        cursor += monom.Structure_.Splits.size();
    }

    offsets.push_back(cursor);

    Features = Buffer<int>::fromVector(flatFeatureIds).data().to(torch::kCUDA);
    Conditions = Buffer<float>::fromVector(conditions).data().to(torch::kCUDA);
    PolynomOffsets = Buffer<int>::fromVector(offsets).data().to(torch::kCUDA);
    PolynomValues = Buffer<float>::fromVector(values).data().to(torch::kCUDA);
}


torch::Tensor PolynomCuda::Forward(torch::Tensor batch) const {
    const int batchSize = batch.size(0);
    int fCount = batch.size(1);

    const int outDim = Polynom_->OutDim();
    const int polynomCount = PolynomOffsets.size(0) - 1;

    torch::Tensor result = torch::zeros({outDim, batchSize},
        TorchHelpers::tensorOptionsOnDevice(ComputeDeviceType::Gpu));
    torch::Tensor probs = torch::zeros({polynomCount, batchSize},
        TorchHelpers::tensorOptionsOnDevice(ComputeDeviceType::Gpu));

    auto transposed = batch.transpose(0, 1);

    PolynomForward(Polynom_->Lambda_,
            transposed.data<float>(),
            fCount,
            batchSize,
            Features.data<int>(),
            Conditions.data<float>(),
            PolynomOffsets.data<int>(),
            PolynomValues.data<float>(),
            polynomCount,
            outDim,
            probs.data<float>(),
            result.data<float>()
        );
    return result.transpose(0, 1);

}
