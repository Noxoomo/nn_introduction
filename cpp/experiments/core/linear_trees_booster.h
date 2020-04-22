#pragma once

#include <methods/boosting.h>
#include <methods/boosting_weak_target_factory.h>
#include <methods/greedy_linear_oblivious_trees.h>
#include <data/grid_builder.h>
#include <data/dataset.h>
#include <util/json.h>

namespace experiments {

struct LinearTreesBoosterOptions {
    BoostingConfig boostingCfg;
    BinarizationConfig binarizationCfg;
    BootstrapOptions boostrapOpts;
    GreedyLinearObliviousTreeLearnerOptions greedyLinearTreesOpts;

    static LinearTreesBoosterOptions fromJson(const json& params);
};

class LinearTreesBooster {
public:
    LinearTreesBooster(const LinearTreesBoosterOptions& opts);

    ModelPtr fit(const DataSet& trainDs, const DataSet& testDs);

    ~LinearTreesBooster() = default;

private:
    LinearTreesBoosterOptions opts_;

};

}
