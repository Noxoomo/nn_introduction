#pragma once

#include "optimizer.h"
#include "listener.h"
#include <memory>
#include <targets/target.h>
#include <models/model.h>
#include <models/ensemble.h>
#include <data/dataset.h>
#include <util/json.h>

struct BoostingConfig {
    double step_ = 0.01;
    int64_t iterations_ = 1000;

    static BoostingConfig fromJson(const json& params);
};

class Boosting : public Optimizer, public ListenersHolder<Model> {
public:

    Boosting(
        const BoostingConfig& config_,
        std::unique_ptr<EmpiricalTargetFactory>&& weak_target_,
        std::unique_ptr<Optimizer>&& weak_learner_);

    ModelPtr fit(const DataSet& dataSet, const Target& target) override;
    ModelPtr fitFrom(std::vector<ModelPtr> models, const DataSet& dataSet, const Target& target);
    ModelPtr fitFrom(std::shared_ptr<Ensemble> ensemble, const DataSet& dataSet, const Target& target);

private:
    BoostingConfig config_;
    std::unique_ptr<EmpiricalTargetFactory> weak_target_;
    std::unique_ptr<Optimizer> weak_learner_;
};
