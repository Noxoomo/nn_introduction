#include <data/dataset.h>
#include <data/load_data.h>

#include <core/vec.h>

#include <gtest/gtest.h>
#include <data/grid_builder.h>
#include <data/binarized_dataset.h>
#include <models/oblivious_tree.h>
#include <vec_tools/transform.h>

#include <models/polynom/polynom.h>
#include <models/polynom/linear_monom.h>

#define EPS 1e-5

//run it from root
TEST(FeaturesTxt, ApplyFloatAndBinarizedOtTest) {
    for (int32_t groupSize : {2, 4, 8, 16, 17, 32}) {
        auto ds = loadFeaturesTxt("test_data/featuresTxt/train");
        EXPECT_EQ(ds.samplesCount(), 12465);
        EXPECT_EQ(ds.featuresCount(), 50);

        BinarizationConfig config;
        config.bordersCount_ = 32;
        auto grid = buildGrid(ds, config);

        for (int32_t i = 0; i < grid->nzFeaturesCount(); ++i) {
            EXPECT_LE(grid->conditionsCount(i), 33);
        }

        auto bds = cachedBinarize(ds, grid, groupSize);

        for (int32_t firstF = 0; firstF < grid->nzFeaturesCount(); firstF += 6) {
            std::vector<BinaryFeature> features;
            for (int32_t i = firstF; i < std::min<int32_t>(firstF + 6, grid->nzFeaturesCount()); ++i) {
                features.emplace_back(i, grid->conditionsCount(i) / 2);
            }
            Vec values(64);
            for (int64_t i = 0; i < 64; ++i) {
                values.arrayRef()[i] = i;
            }

            ObliviousTree tree(grid, features, values);

            Vec toFromBds(ds.samplesCount());
            Vec toFromDs(ds.samplesCount());

            tree.apply(bds, Mx(toFromBds, ds.samplesCount(), 1));
            tree.apply(ds, Mx(toFromDs, ds.samplesCount(), 1));
            EXPECT_EQ(toFromBds, toFromDs);
        }

    }

}

TEST(FeaturesTxt, Gradient) {
    auto ds = loadFeaturesTxt("../../../../test_data/featuresTxt/test");
    std::cout << ds.samplesCount() << std::endl;
    EXPECT_EQ(ds.samplesCount(), 12465);
    EXPECT_EQ(ds.featuresCount(), 50);

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    for (int32_t firstF = 0; firstF < grid->nzFeaturesCount(); firstF += 6) {
        std::vector<BinaryFeature> features;
        for (int32_t i = firstF; i < std::min<int32_t>(firstF + 6, grid->nzFeaturesCount()); ++i) {
            features.emplace_back(i, grid->conditionsCount(i) / 2);
        }
        Vec weg(1 << features.size());
        for (int i = 0; i < weg.dim(); ++i) {
            weg.set(i, 2.0 * std::rand() / RAND_MAX - 1.0 );
        }
        ObliviousTree tree(grid, features, weg);

        double delta = 0.001 / 1000;
        for (int64_t k = 0; k < ds.samplesCount(); ++k) {
            Vec grad = Vec(ds.sample(0).dim());
            tree.grad(ds.sample(k), grad);
            Vec approx = Vec(grad.dim());
            for (int64_t p = 0; p < grad.dim(); ++p) {
                Vec tmp = Vec(grad.dim());
                VecTools::copyTo(ds.sample(k), tmp);
                tmp.set(p, tmp.get(p) + delta);
                approx.set(p, (tree.value(tmp) - tree.value(ds.sample(k))) / delta);
                double appr = approx.get(p);
                double gr = grad.get(p);
                assert(std::abs(appr - gr) <= 0.01 + 0.01*std::abs((gr + appr)/2));
            }
        }
    }
}

TEST(LinearTreeMonom, ValGrad) {

}
