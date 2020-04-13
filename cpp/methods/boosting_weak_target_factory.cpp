#include "boosting_weak_target_factory.h"
#include <core/vec_factory.h>
#include <targets/linear_l2.h>

SharedPtr<Target> GradientBoostingWeakTargetFactory::create(
    const DataSet& ds,
    const Target& target,
    const Mx& startPoint) {
    const Vec cursor = startPoint;
    Vec der(cursor.dim());
    target.gradientTo(cursor, der);
    const auto& l2Target = dynamic_cast<const LinearL2&>(target);
    return std::static_pointer_cast<Target>(std::make_shared<LinearL2>(ds, der, l2Target.l2reg_));
}

template <class Rand>
std::vector<int32_t> uniformBootstrap(int64_t size, double rate, Rand&& rand) {
    std::vector<int32_t> indices;
    for (int64_t i = 0; i < size; ++i) {
        if (rand() <  rate) {
            indices.push_back(i);
        }
    }
    return indices;
}

SharedPtr<Target> GradientBoostingBootstrappedWeakTargetFactory::create(
    const DataSet& ds,
    const Target& target,
    const Mx& startPoint) {
    std::vector<int32_t> nzIndices;
    std::vector<double> nzWeights;

    if (options_.type_ == BootstrapType::Uniform) {
        nzIndices = uniformBootstrap(target.dim(), options_.sampleRate_, [&]() -> double {
            return uniform_(engine_);
        });
        nzWeights.resize(nzIndices.size());
        std::fill(nzWeights.begin(), nzWeights.end(), 1.0);
    } else if(options_.type_ == BootstrapType::Poisson) {
        for (int32_t i = 0; i < target.dim(); ++i) {
            double w =  poisson_(engine_);
            if (w > 0) {
                nzWeights.push_back(w);
                nzIndices.push_back(i);
            }
        }
    } else {
        nzWeights.resize(target.dim());
        nzIndices.resize(target.dim());
        std::iota(nzIndices.begin(), nzIndices.end(), 0);
        for (int64_t i = 0; i < nzWeights.size(); ++i) {
            nzWeights[i] = -log(uniform_(engine_) + 1e-20);
        }
    }

    Vec der(nzWeights.size());
    Vec weights = VecFactory::fromVector(nzWeights);
    Buffer<int32_t> indices = Buffer<int32_t>::fromVector(nzIndices);
    const auto& pointwiseTarget = dynamic_cast<const PointwiseTarget&>(target);
    pointwiseTarget.subsetDer(startPoint, indices, der);
    const auto& l2Target = dynamic_cast<const LinearL2&>(target);
    return std::static_pointer_cast<Target>(std::make_shared<LinearL2>(ds, der, weights, indices, l2Target.l2reg_));
}
