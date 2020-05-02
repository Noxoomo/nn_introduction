#include "boosting_weak_target_factory.h"
#include <core/vec_factory.h>
#include <targets/linear_l2.h>

BootstrapOptions BootstrapOptions::fromJson(const json& params) {
    BootstrapOptions opts;
    opts.sampleRate_ = params.value("sample_rate", opts.sampleRate_);
    if (params.contains("seed")) {
        opts.seed_ = params["seed"];
    }
    std::string type = params.value("type", "poisson");
    if (type == "none") {
        opts.type_ = BootstrapType::None;
    } else if (type == "bayessian") {
        opts.type_ = BootstrapType::Bayessian;
    } else if (type == "uniform") {
        opts.type_ = BootstrapType::Uniform;
    } else if (type == "poisson") {
        opts.type_ = BootstrapType::Poisson;
    }
    return opts;
}

SharedPtr<Target> GradientBoostingWeakTargetFactory::create(
    const DataSet& ds,
    const Target& target,
    const Mx& startPoint) {
    const Vec cursor = startPoint;
    Vec der(cursor.dim());
    target.gradientTo(cursor, der);
    return std::static_pointer_cast<Target>(std::make_shared<LinearL2>(ds, der, l2reg_));
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

    if (options_.type_ == BootstrapType::None) {
        nzWeights.resize(target.dim());
        nzIndices.resize(target.dim());
        std::iota(nzIndices.begin(), nzIndices.end(), 0);
        std::fill(nzWeights.begin(), nzWeights.end(), 1.0);
    } else if (options_.type_ == BootstrapType::Uniform) {
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
        for (uint64_t i = 0; i < nzWeights.size(); ++i) {
            nzWeights[i] = -log(uniform_(engine_) + 1e-20);
        }
    }

    Vec der(nzWeights.size());
    Vec weights = VecFactory::fromVector(nzWeights);
    Buffer<int32_t> indices = Buffer<int32_t>::fromVector(nzIndices);
    const auto& pointwiseTarget = dynamic_cast<const PointwiseTarget&>(target);
    pointwiseTarget.subsetDer(startPoint, indices, der);
    // TODO this god damn params...
    return std::static_pointer_cast<Target>(std::make_shared<LinearL2>(ds, der, weights, indices, l2reg_));
}
