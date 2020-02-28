#include <memory>

#include <data/dataset.h>
#include <data/load_data.h>

#include <gtest/gtest.h>
#include <data/grid_builder.h>
#include <models/oblivious_tree.h>
#include <methods/boosting.h>
#include <methods/greedy_oblivious_tree.h>
#include <methods/greedy_linear_oblivious_trees.h>
#include <methods/greedy_linear_oblivious_trees_v2.h>
#include <methods/boosting_weak_target_factory.h>
#include <targets/cross_entropy.h>
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
        double traceReg,
        GridPtr grid) {
    return std::make_unique<GreedyLinearObliviousTreeLearner>(std::move(grid), depth, biasCol, l2reg, traceReg);
}

inline std::unique_ptr<GreedyLinearObliviousTreeLearnerV2> createWeakLinearLearnerV2(
        int32_t depth,
        int biasCol,
        double l2reg,
        double traceReg,
        GridPtr grid) {
    return std::make_unique<GreedyLinearObliviousTreeLearnerV2>(std::move(grid), depth, biasCol, l2reg, traceReg);
}

inline std::unique_ptr<EmpiricalTargetFactory> createWeakTarget() {
    return std::make_unique<GradientBoostingWeakTargetFactory>();
}

inline std::unique_ptr<EmpiricalTargetFactory>  createBootstrapWeakTarget() {
    BootstrapOptions options;
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

TEST(Boosting, FeaturesTxtLinearTrees) {
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
    Boosting boosting(boostingConfig, createWeakTarget(), createWeakLinearLearner(6, 0, 1.0, 0.01, grid));

    auto testMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
    testMetricsCalcer->addMetric(L2(test), "l2-test");
    boosting.addListener(testMetricsCalcer);

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(ds);
    trainMetricsCalcer->addMetric(L2(ds), "l2-train");
    boosting.addListener(trainMetricsCalcer);

    L2 target(ds);
    auto ensemble = boosting.fit(ds, target);
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

//TEST(HistV2, Simple) {
//    auto ds = simpleDs();
//
//    std::vector<int32_t> indices({0, 1, 2, 3, 4, 5, 6});
//    std::set<int> usedFeatures({});
//
//    BinarizationConfig config;
//    config.bordersCount_ = 32;
//    GridPtr grid = buildGrid(ds, config);
//    BinarizedDataSetPtr bds = binarize(ds, grid, 4);
//
//    std::set<int> features = {0, 1, 2};
//
//    std::cout << "bds.totalBins = " << bds->totalBins() << std::endl;
//
//    HistogramV2 h(*bds, grid, 5, features.size(), 0);
//
//    std::cout << "add bias column" << std::endl;
//    ds.addBiasColumn();
//
//    std::cout << "subDs" << std::endl;
//    auto curDs = ds.subDs(features);
//    std::cout << curDs.samplesMx() << std::endl;
//
//    std::cout << "h.build" << std::endl;
//    h.build(curDs, indices);
//    h.print();
//    std::cout << "h.prefixSumBins" << std::endl;
//    h.prefixSumBins();
//    h.print();
//
//    Vec newCol(ds.samplesCount());
//    ds.copyColumn(4, &newCol);
//    auto newCol_ref = newCol.arrayRef();
//
//    auto ys = ds.target().arrayRef();
//
//    std::cout << "update bins" << std::endl;
//
//    for (int32_t fId = 0; fId < grid->nzFeaturesCount(); ++fId) {
//        bds->visitFeature(fId, indices, [&](int blockId, int i, int8_t localBinId) {
//            Vec x = curDs.sample(i);
//            double fVal = newCol_ref[i];
//            h.updateBin(fId, localBinId, x, ys[i], fVal, 0);
//        });
//    }
//
//    std::cout << "prefix sum last f" << std::endl;
//
//    h.prefixSumBinsLastFeature(0);
//
//    h.print();
//}

TEST(BoostingSimpleV1, V1) {
    auto ds = simpleDs();

    std::vector<int32_t> indices({0, 1, 2, 3, 4, 5, 6});

    ds.addBiasColumn();

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    BoostingConfig boostingConfig;
    boostingConfig.iterations_ = 1;
    boostingConfig.step_ = 1.0;
    Boosting boosting(boostingConfig, createWeakTarget(), createWeakLinearLearner(4, 0, 1e-5, 0.0, grid));

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(ds);
    trainMetricsCalcer->addMetric(L2(ds), "l2-train");
    boosting.addListener(trainMetricsCalcer);

    L2 target(ds);
    auto ensemble = boosting.fit(ds, target);

    for (int i = 0; i < ds.samplesCount(); ++i) {
        std::cout << "y = " << ds.target()(i) << ", y^ = " << ensemble->value(ds.sample(i)) << std::endl;
    }
}

TEST(BoostingSimple, V2) {
    auto ds = simpleDs();

    std::vector<int32_t> indices({0, 1, 2, 3, 4, 5, 6});

    ds.addBiasColumn();

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    BoostingConfig boostingConfig;
    boostingConfig.iterations_ = 1;
    boostingConfig.step_ = 1.0;
    std::cout << "creating boosting" << std::endl;
    Boosting boosting(boostingConfig, createWeakTarget(), createWeakLinearLearnerV2(4, 0, 1e-5, 0.0, grid));
    std::cout << "creating boosting done" << std::endl;

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(ds);
    trainMetricsCalcer->addMetric(L2(ds), "l2-train");
    boosting.addListener(trainMetricsCalcer);

    L2 target(ds);
    auto ensemble = boosting.fit(ds, target);

    for (int i = 0; i < ds.samplesCount(); ++i) {
        std::cout << "y = " << ds.target()(i) << ", y^ = " << ensemble->value(ds.sample(i)) << std::endl;
    }
}

TEST(BoostingSimple, V2Bootstrap) {
    auto ds = simpleDs();

    std::vector<int32_t> indices({0, 1, 2, 3, 4, 5, 6});

    ds.addBiasColumn();

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    BoostingConfig boostingConfig;
    boostingConfig.iterations_ = 10;
    boostingConfig.step_ = 1.0;
    std::cout << "creating boosting" << std::endl;
    Boosting boosting(boostingConfig, createBootstrapWeakTarget(), createWeakLinearLearnerV2(4, 0, 1e-5, 0.0, grid));
    std::cout << "creating boosting done" << std::endl;

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(ds);
    trainMetricsCalcer->addMetric(L2(ds), "l2-train");
    boosting.addListener(trainMetricsCalcer);

    L2 target(ds);
    auto ensemble = boosting.fit(ds, target);

    for (int i = 0; i < ds.samplesCount(); ++i) {
        std::cout << "y = " << ds.target()(i) << ", y^ = " << ensemble->value(ds.sample(i)) << std::endl;
    }
}

TEST(Boosting, LinearV2FeaturesTxt) {
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
    Boosting boosting(boostingConfig, createWeakTarget(), createWeakLinearLearnerV2(6, 0, 0.5, 0.00, grid));

    auto testMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
    testMetricsCalcer->addMetric(L2(test), "l2-test");
    boosting.addListener(testMetricsCalcer);

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(ds);
    trainMetricsCalcer->addMetric(L2(ds), "l2-train");
    boosting.addListener(trainMetricsCalcer);

    L2 target(ds);
    auto ensemble = boosting.fit(ds, target);
}

TEST(Boosting, LinearV2FeaturesTxtBootstrap) {
    auto ds = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/train");
    auto test = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/test");
    EXPECT_EQ(ds.samplesCount(), 12465);
    EXPECT_EQ(ds.featuresCount(), 50);

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    BoostingConfig boostingConfig;
    boostingConfig.iterations_ = 200;
    Boosting boosting(boostingConfig, createBootstrapWeakTarget(), createWeakLinearLearnerV2(6, 0, 0.5, 0.00, grid));

    auto testMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
    testMetricsCalcer->addMetric(L2(test), "l2-test");
    boosting.addListener(testMetricsCalcer);

    auto trainMetricsCalcer = std::make_shared<BoostingMetricsCalcer>(ds);
    trainMetricsCalcer->addMetric(L2(ds), "l2-train");
    boosting.addListener(trainMetricsCalcer);

    L2 target(ds);
    auto ensemble = boosting.fit(ds, target);

}