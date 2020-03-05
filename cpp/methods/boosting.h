#pragma once

#include "optimizer.h"
#include "listener.h"
#include <memory>
#include <models/model.h>
#include <targets/target.h>
#include <models/model.h>
#include <data/dataset.h>

struct BoostingConfig {
    double step_ = 0.01;
    int64_t iterations_ = 1000;

};

class Boosting : public Optimizer, public ListenersHolder<Model> {
public:

    Boosting(
        const BoostingConfig& config_,
        std::unique_ptr<EmpiricalTargetFactory>&& weak_target_,
        std::unique_ptr<Optimizer>&& weak_learner_);

    ModelPtr fit(const DataSet& dataSet, const Target& target) override;

private:
    BoostingConfig config_;
    std::unique_ptr<EmpiricalTargetFactory> weak_target_;
    std::unique_ptr<Optimizer> weak_learner_;
};
