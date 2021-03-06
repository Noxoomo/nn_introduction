#pragma once

#include <methods/boosting.h>
#include <methods/boosting_weak_target_factory.h>
#include <methods/greedy_linear_oblivious_trees.h>
#include <data/grid_builder.h>
#include <data/dataset.h>
#include <util/json.h>

struct LinearTreesBoosterOptions {
    BoostingConfig boostingCfg;
    BinarizationConfig binarizationCfg;
    BootstrapOptions boostrapOpts;
    GreedyLinearObliviousTreeLearnerOptions greedyLinearTreesOpts;

    static LinearTreesBoosterOptions fromJson(const json& params);
};

class LinearTreesBooster {
public:
    explicit LinearTreesBooster(const LinearTreesBoosterOptions& opts);

    ModelPtr fit(const DataSet& trainDs) const;
    ModelPtr fit(const DataSet& trainDs, const DataSet& valDs) const;
    ModelPtr fitFrom(const std::shared_ptr<Ensemble>& ensemble, const DataSet& trainDs, const DataSet& valDs) const;

    ~LinearTreesBooster() = default;

private:
    LinearTreesBoosterOptions opts_;

};
