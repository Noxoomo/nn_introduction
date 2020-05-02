#include "boosting.h"
#include <models/ensemble.h>
#include <chrono>

BoostingConfig BoostingConfig::fromJson(const json& params) {
    BoostingConfig opts;
    opts.step_ = params["step"];
    opts.iterations_ = params["iterations"];
    return opts;
}

ModelPtr Boosting::fit(const DataSet& dataSet, const Target& target)  {
    assert(&dataSet == &target.owner());
    Mx cursor(dataSet.samplesCount(),  1);
    std::vector<ModelPtr> models;

    for (int32_t iter = 0; iter < config_.iterations_; ++iter) {

        auto weakTarget = weak_target_->create(dataSet, target, cursor);

        auto model = weak_learner_->fit(dataSet, *weakTarget);

        model = model->scale(config_.step_);
        models.push_back(model);

        invoke(*models.back());
        models.back()->append(dataSet, cursor);
    }

    return std::make_shared<Ensemble>(std::move(models));
}

Boosting::Boosting(
    const BoostingConfig& config,
    std::unique_ptr<EmpiricalTargetFactory>&& weak_target,
    std::unique_ptr<Optimizer>&& weak_learner)
    : config_(config)
    , weak_target_(std::move(weak_target))
    , weak_learner_(std::move(weak_learner)) {}
