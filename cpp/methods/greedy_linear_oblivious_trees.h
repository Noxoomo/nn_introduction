#pragma once

#include <unordered_set>
#include <vector>
#include <memory>

#include "optimizer.h"

#include <models/model.h>
#include <models/bin_optimized_model.h>
#include <targets/linear_l2_stat.h>
#include <targets/linear_l2.h>
#include <data/grid.h>
#include <core/multi_dim_array.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include <util/json.h>


class LinearObliviousTreeLeafLearner;

struct GreedyLinearObliviousTreeLearnerOptions {
    int maxDepth = 6;
    int biasCol = -1;
    double l2reg = 0.0;

    static GreedyLinearObliviousTreeLearnerOptions fromJson(const json& params);
};

class GreedyLinearObliviousTreeLearner final
        : public Optimizer {
public:
    typedef GreedyLinearObliviousTreeLearnerOptions Options;

    explicit GreedyLinearObliviousTreeLearner(GridPtr grid, Options opts)
            : grid_(std::move(grid))
            , opts_(opts) {
    }

//    GreedyLinearObliviousTreeLearner(const GreedyLinearObliviousTreeLearner& other) = default;

    ModelPtr fit(const DataSet& dataSet, const Target& target) override;

private:
    void cacheDs(const DataSet& ds);

    using TSplit = std::tuple<double, int32_t, int32_t>;

    void buildRoot(const BinarizedDataSet& bds,
                   const DataSet& ds,
                   ConstVecRef<float> ys,
                   ConstVecRef<float> ws);
    void updateNewCorrelations(
            const BinarizedDataSet& bds,
            const DataSet& ds,
            ConstVecRef<float> ys,
            ConstVecRef<float> ws);
    TSplit findBestSplit(const Target& target);
    void initNewLeaves(TSplit split);
    void updateNewLeaves(const BinarizedDataSet& bds,
                         const DataSet& ds,
                         int oldNUsedFeatures,
                         ConstVecRef<float> ys,
                         ConstVecRef<float> ws);
    void updateXs(int origFId);
    float* curX(int sampleId);

    void resetState();
    void resetStats(int nLeaves, int filledSize);

    // TODO add bins factory
    template <typename Stat, typename UpdaterT>
    void ComputeStats(
            int nLeaves, const std::vector<int>& lIds,
            const DataSet& ds, const BinarizedDataSet& bds,
            MultiDimArray<1, MultiDimArray<2, Stat>>& stats,
            UpdaterT updater) {
        int nUsedFeatures = (int)usedFeaturesOrdered_.size();

        // compute stats per [thread Id][leaf Id]
        parallelFor(0, nSamples_, [&](int thId, int sampleId) {
            int origSampleId = indices_[sampleId];
            auto bins = bds.sampleBins(origSampleId);

            int lId = lIds[sampleId];
            if (lId < 0) return;

            auto leafStats = stats[thId][lId];

            for (int fId = 0; fId < fCount_; ++fId) {
                int origFId = grid_->origFeatureIndex(fId);
                int bin = binOffsets_[fId] + bins[fId];
                auto& stat = leafStats[bin];
                updater(stat, sampleId, origFId);
            }
        });

        // gather individual workers results together
        // TODO maybe change order
        parallelFor(0, nLeaves, [&](int lId) {
            auto leftLeafStats = stats[0][lId];
            for (int thId = 1; thId < nThreads_; ++thId) {
                auto rightLeafStats = stats[thId][lId];
                for (int bin = 0; bin < totalBins_; ++bin) {
                    leftLeafStats[bin] += rightLeafStats[bin];
                }
            }
        });

        // prefix sum
        parallelFor(0, fCount_, [&](int fId) {
            int offset = binOffsets_[fId];
            const int condCount = grid_->conditionsCount(fId);
            for (int lId = 0; lId < nLeaves; ++lId) {
                auto leafStats = stats[0][lId];
                for (int bin = 1; bin <= condCount; ++bin) {
                    int absBin = offset + bin;
                    leafStats[absBin] += leafStats[absBin - 1];
                }
            }
        });
    }

private:
    GridPtr grid_;
    Options opts_;

    bool isDsCached_ = false;
    std::vector<Vec> fColumns_;
    std::vector<ConstVecRef<float>> fColumnsRefs_;
    MultiDimArray<2, float> xs_;

    ConstVecRef<int32_t> indices_;

    std::vector<int32_t> leafId_;
    std::vector<std::shared_ptr<LinearObliviousTreeLeafLearner>> leaves_;
    std::vector<std::shared_ptr<LinearObliviousTreeLeafLearner>> newLeaves_;

    std::unique_ptr<MultiDimArray<1, MultiDimArray<2, LinearL2CorStat>>> corStats_;
    std::unique_ptr<MultiDimArray<1, MultiDimArray<2, LinearL2Stat>>> stats_;

    std::set<TSplit> splits_;

    std::set<int> usedFeatures_;
    std::vector<int> usedFeaturesOrdered_;

    std::vector<bool> fullUpdate_;
    std::vector<int> samplesLeavesCnt_;

    std::vector<int> absBinToFId_;

    ConstVecRef<int32_t> binOffsets_;
    int nThreads_;
    int totalBins_;
    int totalCond_;
    int fCount_;
    int nSamples_;
};
