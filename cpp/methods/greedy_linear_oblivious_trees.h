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


class LinearObliviousTreeLeafLearner;

class GreedyLinearObliviousTreeLearner final
        : public Optimizer {
public:
    explicit GreedyLinearObliviousTreeLearner(GridPtr grid, int32_t maxDepth = 6,
            int biasCol = -1, double l2reg = 0.0)
            : grid_(std::move(grid))
            , biasCol_(biasCol)
            , maxDepth_(maxDepth)
            , l2reg_(l2reg) {
    }

    GreedyLinearObliviousTreeLearner(const GreedyLinearObliviousTreeLearner& other) = default;

    ModelPtr fit(const DataSet& dataSet, const Target& target) override;

private:
    void cacheDs(const DataSet& ds);

    using TSplit = std::tuple<double, int32_t, int32_t>;

    void buildRoot(const BinarizedDataSet& bds,
                   const DataSet& ds,
                   ConstVecRef<double> ys,
                   ConstVecRef<double> ws);
    void updateNewCorrelations(
            const BinarizedDataSet& bds,
            const DataSet& ds,
            ConstVecRef<double> ys,
            ConstVecRef<double> ws);
    TSplit findBestSplit(const Target& target);
    void initNewLeaves(TSplit split);
    void updateNewLeaves(const BinarizedDataSet& bds,
                         const DataSet& ds,
                         int oldNUsedFeatures,
                         ConstVecRef<double> ys,
                         ConstVecRef<double> ws);

    void resetState();

    // TODO add bins factory
    template <typename Stat, typename UpdaterT>
    MultiDimArray<2, Stat> ComputeStats(
            int nLeaves, const std::vector<int>& lIds,
            const DataSet& ds, const BinarizedDataSet& bds,
            const Stat& defaultVal,
            UpdaterT updater) {
        int nUsedFeatures = usedFeaturesOrdered_.size();

        MultiDimArray<1, MultiDimArray<2, Stat>> stats({nThreads_});
        MultiDimArray<1, std::vector<double>> curX({nThreads_});

        parallelFor(0, nThreads_, [&](int thId) {
            curX[thId] = std::vector<double>(nUsedFeatures, 0.);
            stats[thId] = MultiDimArray<2, Stat>({nLeaves, totalBins_}, defaultVal);
        });

        // compute stats per [thread Id][leaf Id]
        parallelFor(0, nSamples_, [&](int thId, int sampleId) {
            auto& x = curX[thId];
            auto bins = bds.sampleBins(sampleId);
            int lId = lIds[sampleId];
            if (lId < 0) return;
            auto leafStats = stats[thId][lId];

            ds.fillSample(sampleId, usedFeaturesOrdered_, x);

            for (int fId = 0; fId < fCount_; ++fId) {
                int bin = binOffsets_[fId] + bins[fId];
                auto& stat = leafStats[bin];
                updater(stat, x, sampleId, fId);
            }
        });

        // gather individual workers results together
        // TODO maybe change order
        parallelFor(0, nLeaves, [&](int lId) {
            for (int thId = 1; thId < nThreads_; ++thId) {
                for (int bin = 0; bin < totalBins_; ++bin) {
                    stats[0][lId][bin] += stats[thId][lId][bin];
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

        // TODO this copies can affect performance. Need to rework MultiDimArray a bit
        return stats[0].copy();
    }

private:
    GridPtr grid_;
    int32_t maxDepth_ = 6;
    int biasCol_ = -1;
    double l2reg_ = 0.0;

    bool isDsCached_ = false;
    std::vector<Vec> fColumns_;
    std::vector<ConstVecRef<double>> fColumnsRefs_;

    std::vector<int32_t> leafId_;
    std::vector<std::shared_ptr<LinearObliviousTreeLeafLearner>> leaves_;
    std::vector<std::shared_ptr<LinearObliviousTreeLeafLearner>> newLeaves_;

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
