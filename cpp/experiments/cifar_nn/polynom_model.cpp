#include "polynom_model.h"
#include <core/torch_helpers.h>

torch::Tensor PolynomModel::forward(torch::Tensor samples){
    VERIFY(polynom_, "set polynom first");
    const int batchSize = samples.size(0);
    auto yDim = TorchHelpers::totalSize(samples) / batchSize;
    samples = samples.reshape({batchSize, yDim});
    auto samplesDevice = samples.device();
    if (samplesDevice == torch::kCPU) {
        return PolynomForward(polynom_).apply({samples})[0];
    }
    else {
        return PolynomForwardGPU(polynom_).apply({samples})[0];
    }
}
