#pragma once

#include <core/object.h>
#include <core/matrix.h>
#include <data/dataset.h>
#include <targets/target.h>
#include <targets/l2.h>
#include <random>
#include <util/json.h>

class GradientBoostingWeakTargetFactory : public EmpiricalTargetFactory {
public:
    virtual SharedPtr<Target> create(const DataSet& ds,
                                     const Target& target,
                                     const Mx& startPoint)  override;
private:
//    bool UseNewtonForC2 = false;
};



enum class BootstrapType {
    None,
    Bayessian,
    Uniform,
    Poisson
};

struct BootstrapOptions {
    BootstrapType type_ = BootstrapType::Poisson;
    double sampleRate_ = 0.7;
    uint32_t seed_ = 42;

    static BootstrapOptions fromJson(const json& params);
};

class GradientBoostingBootstrappedWeakTargetFactory : public EmpiricalTargetFactory {
public:
    GradientBoostingBootstrappedWeakTargetFactory(BootstrapOptions options)
    : options_(std::move(options))
    , engine_(options_.seed_) {

    }

    virtual SharedPtr<Target> create(const DataSet& ds,
                                     const Target& target,
                                     const Mx& startPoint) override;
private:
    BootstrapOptions options_;
    std::default_random_engine engine_;
    std::uniform_real_distribution<double> uniform_ = std::uniform_real_distribution<double>(0, 1);
    std::poisson_distribution<int> poisson_ = std::poisson_distribution<int>(1);


};
