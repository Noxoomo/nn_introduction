#include "polynom_gpu_autograd.h"
#include <core/vec.h>
#include <vec_tools/fill.h>
#include <util/array_ref.h>

torch::autograd::variable_list PolynomBackwardGPU::apply(torch::autograd::variable_list&& inputs)  {
  auto dims = samplesBatch_.sizes();
  const auto batchSize = dims[0];
  const auto featuresCount = dims[1];

  torch::Tensor grads = torch::zeros({batchSize, featuresCount}, torch::kFloat32);
  auto backGrads = inputs[0];

  return {polynom_->Backward(samplesBatch_, backGrads)};

}


torch::autograd::variable_list PolynomForwardGPU::apply(torch::autograd::variable_list&& inputs) {
    torch::autograd::Variable samplesBatch = inputs[0];

    auto dims = samplesBatch.sizes();
    const int batchSize = dims[0];
    const int outDim = polynom_->Polynom_->OutDim();
    VERIFY(outDim > 0, "Error");
    torch::autograd::Variable result = polynom_->Forward(samplesBatch);

    auto gradFunc = std::make_shared<PolynomBackwardGPU>(samplesBatch,
                                                      polynom_,
                                                      torch::autograd::collect_next_edges(inputs));

    torch::autograd::create_gradient_edge(result,
                                          gradFunc);

    return {result};
}
