#include "linear_trees_booster.h"

#include <targets/linear_l2.h>
#include <methods/greedy_linear_oblivious_trees.h>

namespace experiments {

// TODO move it somewhere
class BinaryAcc : public Stub<Func, BinaryAcc> {
public:
    explicit BinaryAcc(const DataSet& ds, double threshold = 0.5)
            : Stub<Func, BinaryAcc>(ds.featuresCount())
            , ds_(ds)
            , threshold_(threshold) {

    }

    void valueTo(const Vec& vec, double& res) const {
        // TODO maybe a better way with libtorch to do this? Like torch::where

        auto vecRef = vec.arrayRef();
        auto targetRef = ds_.target().arrayRef();

        int correct = 0;

        for (int i = 0; i < targetRef.size(); ++i) {
            double val = 0.0;
            if (vecRef[i] > threshold_) {
                val = 1.0;
            }
            std::cout << std::setw(6) << vecRef[i] << " " << targetRef[i] << std::endl;

            if (std::abs(val - targetRef[i]) < 1e-8) {
                correct++;
            }
        }

        res = 1 - correct * 1.0 / targetRef.size();
    }

private:
    const DataSet& ds_;
    double threshold_;
};

LinearTreesBoosterOptions LinearTreesBoosterOptions::fromJson(const json& params) {
    LinearTreesBoosterOptions opts;

    opts.binarizationCfg = BinarizationConfig::fromJson(params["binarization"]);
    opts.boostingCfg = BoostingConfig::fromJson(params["boosting"]);
    opts.boostrapOpts = BootstrapOptions::fromJson(params["bootstrap"]);
    opts.greedyLinearTreesOpts = GreedyLinearObliviousTreeLearnerOptions::fromJson(params["linear_trees"]);

    return opts;
}

static std::unique_ptr<GreedyLinearObliviousTreeLearner> createWeakLinearLearner(
        GridPtr grid,
        const GreedyLinearObliviousTreeLearnerOptions& opts) {
    return std::make_unique<GreedyLinearObliviousTreeLearner>(std::move(grid), opts);
}

static std::unique_ptr<EmpiricalTargetFactory> createBootstrapWeakTarget(BootstrapOptions opts) {
    return std::make_unique<GradientBoostingBootstrappedWeakTargetFactory>(opts);
}

LinearTreesBooster::LinearTreesBooster(const LinearTreesBoosterOptions& opts)
        : opts_(opts) {

}

ModelPtr LinearTreesBooster::fit(const DataSet& trainDs, const DataSet& testDs) {
    auto grid = buildGrid(trainDs, opts_.binarizationCfg);

    Boosting boosting(opts_.boostingCfg,
                      createBootstrapWeakTarget(opts_.boostrapOpts),
                      createWeakLinearLearner(grid, opts_.greedyLinearTreesOpts));

    auto testMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(testDs);
    testMetricsCalcer->addMetric(L2(testDs), "l2-test");
    testMetricsCalcer->addMetric(BinaryAcc(testDs), "l2-acc", 10);
    boosting.addListener(testMetricsCalcer);

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(trainDs);
    trainMetricsCalcer->addMetric(L2(trainDs), "l2-train");
    boosting.addListener(trainMetricsCalcer);

    auto fitTimeCalcer = std::make_shared<BoostingFitTimeTracker>();
    boosting.addListener(fitTimeCalcer);

    LinearL2 target(trainDs, opts_.greedyLinearTreesOpts.l2reg);
    auto ensemble = boosting.fit(trainDs, target);
}

}
