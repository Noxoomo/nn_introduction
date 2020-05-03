#include "boosting.h"
#include <models/ensemble.h>
#include <chrono>

BoostingConfig BoostingConfig::fromJson(const json& params) {
    BoostingConfig opts;
    opts.step_ = params.value("step", opts.step_);
    opts.iterations_ = params.value("iterations", opts.iterations_);
    return opts;
}

ModelPtr Boosting::fit(const DataSet& dataSet, const Target& target)  {
    return fitFrom(std::vector<ModelPtr>(), dataSet, target);
}

ModelPtr Boosting::fitFrom(std::vector<ModelPtr> models, const DataSet& dataSet, const Target& target)  {
    assert(&dataSet == &target.owner());
    Mx cursor(dataSet.samplesCount(),  1);

    int64_t iter = 0;
    for (; iter < (int64_t)models.size(); ++iter) {
        invoke(*models.back());
        models.back()->append(dataSet, cursor);
    }

    for (; iter < config_.iterations_; ++iter) {
        auto weakTarget = weak_target_->create(dataSet, target, cursor);

        auto model = weak_learner_->fit(dataSet, *weakTarget);

        model = model->scale(config_.step_);
        models.push_back(model);

        invoke(*models.back());
        models.back()->append(dataSet, cursor);
    }

    return std::make_shared<Ensemble>(std::move(models));
}

ModelPtr Boosting::fitFrom(std::shared_ptr<Ensemble> ensemble, const DataSet& dataSet, const Target& target)  {
    assert(&dataSet == &target.owner());
    std::vector<ModelPtr> models;
    ensemble->visitModels([&](ModelPtr model) {
        models.emplace_back(std::move(model));
    });
    return fitFrom(models, dataSet, target);
}

Boosting::Boosting(
    const BoostingConfig& config,
    std::unique_ptr<EmpiricalTargetFactory>&& weak_target,
    std::unique_ptr<Optimizer>&& weak_learner)
    : config_(config)
    , weak_target_(std::move(weak_target))
    , weak_learner_(std::move(weak_learner)) {}
