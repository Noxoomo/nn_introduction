#include <memory>
#include <stdlib.h>
#include <time.h>
#include <random>

#include <data/dataset.h>
#include <data/load_data.h>

#include <gtest/gtest.h>
#include <data/grid_builder.h>
#include <models/oblivious_tree.h>
#include <methods/boosting.h>
#include <methods/greedy_oblivious_tree.h>
#include <methods/greedy_linear_oblivious_trees.h>
#include <methods/boosting_weak_target_factory.h>
#include <targets/cross_entropy.h>
#include <targets/linear_l2.h>
#include <metrics/accuracy.h>

#define EPS 1e-5
#define PATH_PREFIX "../../../../"

inline std::unique_ptr<GreedyObliviousTree> createWeakLearner(
    int32_t depth,
    GridPtr grid) {
    return std::make_unique<GreedyObliviousTree>(std::move(grid), depth);
}

inline std::unique_ptr<GreedyLinearObliviousTreeLearner> createWeakLinearLearner(
        int32_t depth,
        int biasCol,
        double l2reg,
        GridPtr grid) {
    return std::make_unique<GreedyLinearObliviousTreeLearner>(std::move(grid), depth, biasCol, l2reg);
}

inline std::unique_ptr<EmpiricalTargetFactory> createWeakTarget() {
    return std::make_unique<GradientBoostingWeakTargetFactory>();
}

inline std::unique_ptr<EmpiricalTargetFactory> createBootstrapWeakTarget() {
    BootstrapOptions options;
//    srand(time(NULL));
//    options.seed_ = std::rand() % 10000;
    options.seed_ = 42;
    return std::make_unique<GradientBoostingBootstrappedWeakTargetFactory>(options);
}

TEST(FeaturesTxt, TestTrainMseFeaturesTxt) {
    auto ds = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/train");
    auto test = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/test");
    EXPECT_EQ(ds.samplesCount(), 12465);
    EXPECT_EQ(ds.featuresCount(), 50);

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    BoostingConfig boostingConfig;
    Boosting boosting(boostingConfig, createWeakTarget(), createWeakLearner(6, grid));

    auto metricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
    metricsCalcer->addMetric(L2(test), "l2");
    boosting.addListener(metricsCalcer);
    L2 target(ds);
    auto ensemble = boosting.fit(ds, target);
}


TEST(FeaturesTxt, TestTrainWithBootstrapMseFeaturesTxt) {
    auto ds = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/train");
    auto test = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/test");
    EXPECT_EQ(ds.samplesCount(), 12465);
    EXPECT_EQ(ds.featuresCount(), 50);

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    BoostingConfig boostingConfig;
    Boosting boosting(boostingConfig, createBootstrapWeakTarget(), createWeakLearner(6, grid));

    auto metricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
    metricsCalcer->addMetric(L2(test), "l2");
    boosting.addListener(metricsCalcer);
    L2 target(ds);
    auto ensemble = boosting.fit(ds, target);

}

TEST(FeaturesTxt, TestTrainWithBootstrapLogLikelihoodFeaturesTxt) {
    auto ds = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/train");

    auto test = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/test");
    EXPECT_EQ(ds.samplesCount(), 12465);
    EXPECT_EQ(ds.featuresCount(), 50);

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    BoostingConfig boostingConfig;
//    Boosting boosting(boostingConfig, createBootstrapWeakTarget(), createWeakLearner(6, grid));
    Boosting boosting(boostingConfig, createWeakTarget(), createWeakLearner(6, grid));

    auto metricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
    metricsCalcer->addMetric(CrossEntropy(test, 0.1), "CrossEntropy");
    metricsCalcer->addMetric(Accuracy(test.target(), 0.1, 0), "Accuracy");
    boosting.addListener(metricsCalcer);
    CrossEntropy target(ds, 0.1);
    auto ensemble = boosting.fit(ds, target);

}

//run it from root
TEST(FeaturesTxt, TestTrainMseMoscow) {
    auto start = std::chrono::system_clock::now();

    auto ds = loadFeaturesTxt("/Users/noxoomo/Projects/moscow_learn_200k.tsv");
    auto test = loadFeaturesTxt("/Users/noxoomo/Projects/moscow_test.tsv");
//    auto ds = loadFeaturesTxt("moscow_learn_200k.tsv");
//    auto test = loadFeaturesTxt("moscow_test.tsv");

    std::cout << " load data in memory " << std::endl;
    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);
    std::cout << " build grid " << std::endl;

    BoostingConfig boostingConfig;
    Boosting boosting(boostingConfig, createBootstrapWeakTarget(), createWeakLearner(6, grid));

    auto metricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
    metricsCalcer->addMetric(L2(test), "l2");
    boosting.addListener(metricsCalcer);
    L2 target(ds);
    auto ensemble = boosting.fit(ds, target);
    std::cout << "total time " << std::chrono::duration<double>(std::chrono::system_clock::now() - start).count()
              << std::endl;

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

TEST(BoostingLinearTrees, SimpleDs) {
    auto ds = simpleDs();

    std::vector<int32_t> indices({0, 1, 2, 3, 4, 5, 6});

    ds.addBiasColumn();

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    BoostingConfig boostingConfig;
    boostingConfig.iterations_ = 1;
    boostingConfig.step_ = 1;
    Boosting boosting(boostingConfig, createWeakTarget(), createWeakLinearLearner(6, 0, 1e-5, grid));

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(ds);
    trainMetricsCalcer->addMetric(L2(ds), "l2-train");
    boosting.addListener(trainMetricsCalcer);

    LinearL2 target(ds);
    auto ensemble = boosting.fit(ds, target);

    for (int i = 0; i < ds.samplesCount(); ++i) {
        std::cout << "y = " << ds.target()(i) << ", y^ = " << ensemble->value(ds.sample(i)) << std::endl;
    }
}

TEST(BoostingLinearTrees, FeaturesTxt) {
    auto ds = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/train");
    auto test = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/test");
    EXPECT_EQ(ds.samplesCount(), 12465);
    EXPECT_EQ(ds.featuresCount(), 50);

    ds.addBiasColumn();
    test.addBiasColumn();

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    BoostingConfig boostingConfig;
    boostingConfig.iterations_ = 1000;
    boostingConfig.step_ = 0.05;
    Boosting boosting(boostingConfig, createWeakTarget(), createWeakLinearLearner(6, 0, 1., grid));

    auto testMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
    testMetricsCalcer->addMetric(L2(test), "l2-test");
    boosting.addListener(testMetricsCalcer);

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(ds);
    trainMetricsCalcer->addMetric(L2(ds), "l2-train");
    boosting.addListener(trainMetricsCalcer);

    LinearL2 target(ds);
    auto ensemble = boosting.fit(ds, target);
}

TEST(BoostingLinearTrees, FeaturesTxtBootsrap) {
    auto ds = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/train");
    auto test = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/test");
    EXPECT_EQ(ds.samplesCount(), 12465);
    EXPECT_EQ(ds.featuresCount(), 50);

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    BoostingConfig boostingConfig;
    boostingConfig.iterations_ = 500;
    boostingConfig.step_ = 0.05;
    Boosting boosting(boostingConfig, createBootstrapWeakTarget(), createWeakLinearLearner(6, 0, 50.0, grid));

    auto testMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
    testMetricsCalcer->addMetric(L2(test), "l2-test");
    boosting.addListener(testMetricsCalcer);

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(ds);
    trainMetricsCalcer->addMetric(L2(ds), "l2-train");
    boosting.addListener(trainMetricsCalcer);

    LinearL2 target(ds);
    auto ensemble = boosting.fit(ds, target);
}
