#include "polynom_model.h"
#include <core/torch_helpers.h>
#include <models/polynom/polynom_gpu_autograd.h>
#include <models/polynom/polynom_gpu.h>

torch::Tensor PolynomModel::forward(torch::Tensor samples){
    VERIFY(polynom_, "set polynom first");
    auto polynomForward = PolynomForward(polynom_);
    const int batchSize = samples.size(0);
    auto yDim = TorchHelpers::totalSize(samples) / batchSize;
    samples = samples.reshape({batchSize, yDim});
    auto samplesDevice = samples.device();
    if (true) {
         return PolynomForwardGPU(std::make_shared<PolynomCuda>(polynom_)).apply({samples})[0];
    }
    samples = samples.to(torch::kCPU);
    return polynomForward.apply({samples})[0].to(samplesDevice);
}
