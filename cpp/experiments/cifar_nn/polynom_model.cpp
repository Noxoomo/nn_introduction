#include "polynom_model.h"
#include <core/torch_helpers.h>

torch::Tensor PolynomModel::forward(torch::Tensor samples){
    VERIFY(polynom_, "set polynom first");
    auto polynomForward = PolynomForward(polynom_);
    const int batchSize = samples.size(0);
    auto yDim = TorchHelpers::totalSize(samples) / batchSize;
    samples = samples.reshape({batchSize, yDim});
    auto samplesDevice = samples.device();
    //TODO: looks like dirty hack
    samples = samples.to(torch::kCPU);
    return polynomForward.apply({samples})[0].to(samplesDevice);
}
