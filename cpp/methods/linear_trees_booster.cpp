#include "linear_trees_booster.h"

#include <targets/cross_entropy.h>
#include <methods/greedy_linear_oblivious_trees.h>
#include <vec_tools/transform.h>
#include <methods/greedy_oblivious_tree.h>
#include <models/polynom/polynom.h>
#include <models/polynom/linear_monom.h>
#include <experiments/core/polynom_model.h>


// TODO move it somewhere
class BinaryAcc : public Stub<Func, BinaryAcc> {
public:
    explicit BinaryAcc(const DataSet& ds, double threshold = 0.0)
            : Stub<Func, BinaryAcc>(ds.featuresCount())
            , ds_(ds)
            , threshold_(threshold) {

    }

    void valueTo(const Vec& x, double& res) const {
        // TODO maybe a better way with libtorch to do this? Like torch::where

        auto xRef = x.arrayRef();
        auto targetRef = ds_.target().arrayRef();

        int correct = 0;

        for (uint64_t i = 0; i < targetRef.size(); ++i) {
            double val = 0.0;
            if (xRef[i] > threshold_) {
                val = 1.0;
            }

            if (std::abs(val - targetRef[i]) < 1e-8) {
                correct++;
            }
        }

        res = correct * 1.0 / targetRef.size();
    }

private:
    const DataSet& ds_;
    double threshold_;

};

LinearTreesBoosterOptions LinearTreesBoosterOptions::fromJson(const json& params) {
    LinearTreesBoosterOptions opts;

    opts.binarizationCfg = BinarizationConfig::fromJson(params["grid_config"]);
    opts.boostingCfg = BoostingConfig::fromJson(params["boosting_config"]);
    opts.boostrapOpts = BootstrapOptions::fromJson(params["bootstrap_options"]);
    opts.greedyLinearTreesOpts = GreedyLinearObliviousTreeLearnerOptions::fromJson(params["tree_config"]);

    return opts;
}

static std::unique_ptr<GreedyLinearObliviousTreeLearner> createWeakLinearLearner(
        GridPtr grid,
        const GreedyLinearObliviousTreeLearnerOptions& opts) {
    return std::make_unique<GreedyLinearObliviousTreeLearner>(std::move(grid), opts);
}

static std::unique_ptr<EmpiricalTargetFactory> createBootstrapWeakTarget(
        BootstrapOptions opts, double l2reg) {
    return std::make_unique<GradientBoostingBootstrappedWeakTargetFactory>(opts, l2reg);
}

LinearTreesBooster::LinearTreesBooster(const LinearTreesBoosterOptions& opts)
        : opts_(opts) {

}

ModelPtr LinearTreesBooster::fit(const DataSet& trainDs) const {
    auto grid = buildGrid(trainDs, opts_.binarizationCfg);

    Boosting boosting(opts_.boostingCfg,
                      createBootstrapWeakTarget(opts_.boostrapOpts, opts_.greedyLinearTreesOpts.l2reg),
                      createWeakLinearLearner(grid, opts_.greedyLinearTreesOpts));

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(trainDs);
    trainMetricsCalcer->addMetric(CrossEntropy(trainDs), "cross_entropy-train", 1, BoostingMetricsCalcer::MetricType::Maximization);
    trainMetricsCalcer->addMetric(BinaryAcc(trainDs), "acc-train", 1, BoostingMetricsCalcer::MetricType::Maximization);
    boosting.addListener(trainMetricsCalcer);

    auto fitTimeCalcer = std::make_shared<BoostingFitTimeTracker>();
    boosting.addListener(fitTimeCalcer);

    CrossEntropy target(trainDs);
    auto ensemble = boosting.fit(trainDs, target);

    return ensemble;
}

ModelPtr LinearTreesBooster::fit(const DataSet& trainDs, const DataSet& valDs) const {
    auto grid = buildGrid(trainDs, opts_.binarizationCfg);

    Boosting boosting(opts_.boostingCfg,
                      createBootstrapWeakTarget(opts_.boostrapOpts, opts_.greedyLinearTreesOpts.l2reg),
                      createWeakLinearLearner(grid, opts_.greedyLinearTreesOpts));

    auto testMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(valDs);
    testMetricsCalcer->addMetric(CrossEntropy(valDs), "cross_entropy-val", 1, BoostingMetricsCalcer::MetricType::Maximization);
    testMetricsCalcer->addMetric(BinaryAcc(valDs), "acc-val", 1, BoostingMetricsCalcer::MetricType::Maximization);
    boosting.addListener(testMetricsCalcer);

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(trainDs);
    trainMetricsCalcer->addMetric(CrossEntropy(trainDs), "cross_entropy-train", 1, BoostingMetricsCalcer::MetricType::Maximization);
    trainMetricsCalcer->addMetric(BinaryAcc(trainDs), "acc-train", 1, BoostingMetricsCalcer::MetricType::Maximization);
    boosting.addListener(trainMetricsCalcer);

    auto fitTimeCalcer = std::make_shared<BoostingFitTimeTracker>();
    boosting.addListener(fitTimeCalcer);

    CrossEntropy target(trainDs);
    auto ensemble = boosting.fit(trainDs, target);

    /*
    Mx cursor1(valDs.samplesCount(), 1);
    {
        ensemble->apply(valDs, cursor1);
        std::cout << "ensemble size: " << std::dynamic_pointer_cast<Ensemble>(ensemble)->size() << std::endl;
        std::dynamic_pointer_cast<Ensemble>(ensemble)->visitModels([&](ModelPtr model) {
            auto cmodel = std::dynamic_pointer_cast<LinearObliviousTree>(model);
            cmodel->printInfo();
        });
        std::cout << "ensemble acc: " << std::setprecision(5) << BinaryAcc(valDs).value(cursor1) << std::endl;
    }

    {
        auto polynom = std::make_shared<Polynom>(LinearTreesToPolynom(*std::dynamic_pointer_cast<Ensemble>(ensemble)));
        std::cout << "polynom size: " << polynom->Ensemble_.size() << std::endl;
        std::cout << *polynom << std::endl;
        auto polynomModel = std::make_shared<PolynomModel>(Monom::MonomType::LinearMonom);
        polynomModel->reset(polynom);
        auto tIdxs = torch::ones({1}, torch::kLong);
        auto res = polynomModel->forward(valDs.tensorData().view({valDs.samplesCount(), -1})).index_select(1, tIdxs);
        Vec cursor(res.view({-1}));
        std::cout << "polynom acc: " << std::setprecision(5) << BinaryAcc(valDs).value(cursor) << std::endl;

        float mxdiff = 0;
        int idx;
        for (int i = 0; i < cursor.size(); ++i) {
            if (std::abs(cursor(i) - cursor1.get(i,0).value()) > mxdiff) {
                idx = i;
                mxdiff = std::abs(cursor(i) - cursor1.get(i,0).value())
            }
        }

        std::cout << "mxdiff = " << mxdiff << std::endl;
        std::cout << "ensemble val: " << cursor1.get(idx, 0).value() << ", poly val: " << cursor(idx) << std::endl;

        auto sample = valDs.sample(idx).data();

        for (int i = 0; i < valDs.featuresCount(); ++i) {
            std::cout << "f[" << i << "]: " << sample[i] << std::endl;
        }

        exit(1);
    }
     */

    return ensemble;
}
