#include <gtest/gtest.h>

#include <iostream>

#include <models/linear_oblivious_tree.h>
#include <models/polynom/linear_monom.h>
#include <models/polynom/polynom.h>
#include <data/dataset.h>
#include <data/grid_builder.h>
#include <core/vec.h>
#include <core/vec_factory.h>
#include <methods/boosting.h>
#include <methods/listener.h>
#include <methods/greedy_linear_oblivious_trees.h>
#include <methods/boosting_weak_target_factory.h>
#include <models/polynom/polynom_gpu.h>


inline std::unique_ptr<GreedyLinearObliviousTreeLearner> createWeakLinearLearner(
        int32_t depth,
        int biasCol,
        double l2reg,
        GridPtr grid) {
    GreedyLinearObliviousTreeLearnerOptions opts;
    opts.biasCol = biasCol;
    opts.maxDepth = depth;
    opts.l2reg = l2reg;
    return std::make_unique<GreedyLinearObliviousTreeLearner>(std::move(grid), opts);
}

inline std::unique_ptr<EmpiricalTargetFactory> createWeakTarget() {
    return std::make_unique<GradientBoostingWeakTargetFactory>();
}

DataSet simpleDs() {
    Vec dsDataVec = VecFactory::fromVector({
                                                   0.1,   0,     0, 1.1,
                                                   2,     0.1,   0, 1.2,
                                                   3,     -17,   0, 1.3,
                                                   4,     1,     0, 1.4,
                                                   5,     .2,    0, 1.5,
                                                   6,     .1337, 0, 1.6,
                                                   8,     2.17,  0, 1.7,
                                           });
    Vec target = VecFactory::fromVector({
                                                1,
                                                3.5,
                                                3.9,
                                                0,
                                                -6,
                                                -6.8,
                                                -9.1,
                                        });

    return DataSet(Mx(dsDataVec, 7, 4), target);
}

TEST(LinearPolynom, ValGrad) {
    auto ds = simpleDs();

    std::vector<int32_t> indices({0, 1, 2, 3, 4, 5, 6});

    ds.addBiasColumn();

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    const double l2reg = 1e-5;

    BoostingConfig boostingConfig;
    boostingConfig.iterations_ = 3;
    boostingConfig.step_ = 0.5;
    Boosting boosting(boostingConfig, createWeakTarget(), createWeakLinearLearner(3, 0, l2reg, grid));

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(ds);
    trainMetricsCalcer->addMetric(L2(ds), "l2-train");
    boosting.addListener(trainMetricsCalcer);

    LinearL2 target(ds, l2reg);
    auto ensemble = boosting.fit(ds, target);

    auto polynom = LinearTreesToPolynom(*std::dynamic_pointer_cast<Ensemble>(ensemble));

    for (int i = 0; i < ds.samplesCount(); ++i) {
        Vec val(1);
        polynom.Forward(ds.sample(i).arrayRef(), val.arrayRef());
        std::cout << "val for sample #" << i << ": " << ensemble->value(ds.sample(i)) << " " << val(0) << std::endl;
        ASSERT_NEAR(ensemble->value(ds.sample(i)), val(0), 1e-4);
    }

    for (int i = 0; i < ds.samplesCount(); ++i) {
        Vec gradExpected(ds.featuresCount());
        ensemble->grad(ds.sample(i), gradExpected);

        Vec gradActual(ds.featuresCount());
        auto outputGrads = VecFactory::fromVector({1});
        polynom.Backward(ds.sample(i).arrayRef(), outputGrads.arrayRef(), gradActual.arrayRef());

        for (int j = 0; j < gradExpected.size(); ++j) {
            std::cout << "grad for sample #" << i << ": " << gradExpected(j) << " " << gradActual(j) << std::endl;
            ASSERT_NEAR(gradExpected(j), gradActual(j), 1e-4);
        }
    }
}

TEST(LinearPolynomGpu, ValGrad) {
    auto ds = simpleDs();

    std::vector<int32_t> indices({0, 1, 2, 3, 4, 5, 6});

    ds.addBiasColumn();

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    const double l2reg = 1e-5;

    BoostingConfig boostingConfig;
    boostingConfig.iterations_ = 3;
    boostingConfig.step_ = 0.5;
    Boosting boosting(boostingConfig, createWeakTarget(), createWeakLinearLearner(3, 0, l2reg, grid));

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(ds);
    trainMetricsCalcer->addMetric(L2(ds), "l2-train");
    boosting.addListener(trainMetricsCalcer);

    LinearL2 target(ds, l2reg);
    auto ensemble = boosting.fit(ds, target);

    auto polynom = std::make_shared<Polynom>(LinearTreesToPolynom(*std::dynamic_pointer_cast<Ensemble>(ensemble)));
    auto gpuPolynom = PolynomCuda(polynom);

    // TODO why `/ 2`???
    for (int i = 0; i < ds.samplesCount(); ++i) {
        auto val = gpuPolynom.Forward(ds.sample(i).data().view({1, -1}).to(torch::kCUDA)).to(torch::kCPU);
        std::cout << "val for sample #" << i << ": " << ensemble->value(ds.sample(i)) << " " << val.data_ptr<float>()[0] << std::endl;
        ASSERT_NEAR(ensemble->value(ds.sample(i)), val.data_ptr<float>()[0] / 2, 1e-4);
    }

    for (int i = 0; i < ds.samplesCount(); ++i) {
        Vec gradExpected(ds.featuresCount());
        ensemble->grad(ds.sample(i), gradExpected);

        auto outputGrads = VecFactory::fromVector({1});
        auto gradActual = gpuPolynom.Backward(ds.sample(i).data().view({1, -1}).to(torch::kCUDA),
                outputGrads.data().view({1, -1})).to(torch::kCPU);

        for (int j = 0; j < gradExpected.size(); ++j) {
            std::cout << "grad for sample #" << i << ": " << gradExpected(j) << " " << gradActual[0][j].data_ptr<float>()[0] << std::endl;
            ASSERT_NEAR(gradExpected(j), gradActual[0][j].data_ptr<float>()[0], 1e-4);
        }
    }
}
