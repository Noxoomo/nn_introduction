#include "polynom_autograd.h"
#include <core/vec.h>
#include <vec_tools/fill.h>
#include <util/array_ref.h>

torch::autograd::variable_list PolynomBackward::apply(torch::autograd::variable_list&& inputs)  {
  auto dims = samplesBatch_.sizes();
  const auto batchSize = dims[0];
  const auto featuresCount = dims[1];

  torch::Tensor grads = torch::zeros({batchSize, featuresCount}, torch::kFloat32);
  auto backGrads = inputs[0];

  parallelFor(0, batchSize, [&](int64_t i) {
    Vec sample = Vec(samplesBatch_[i]);
    Vec backDers = Vec(backGrads[i]);
    Vec featureDers = Vec(grads[i]);
    polynom_->Backward(sample.arrayRef(), backDers.arrayRef(), featureDers.arrayRef());
  });
  return {grads};
}


torch::autograd::variable_list PolynomForward::apply(torch::autograd::variable_list&& inputs) {
    torch::autograd::Variable samplesBatch = inputs[0];

    auto dims = samplesBatch.sizes();
    const int batchSize = dims[0];
    const int outDim = polynom_->OutDim();
    VERIFY(outDim > 0, "Error");
    torch::autograd::Variable result = torch::zeros({batchSize, outDim}, torch::kFloat32);

    for (int i = 0; i < batchSize; ++i) {
        Vec resultVec = Vec(result[i]);
        VecTools::fill(0.0f, resultVec);
        auto sample = Vec(samplesBatch[i]);
        polynom_->Forward(sample.arrayRef(), resultVec.arrayRef());
    }

    auto gradFunc = std::make_shared<PolynomBackward>(samplesBatch,
                                                      polynom_,
                                                      torch::autograd::collect_next_edges(inputs));

    torch::autograd::create_gradient_edge(result,
                                          gradFunc);

    return {result};
}
