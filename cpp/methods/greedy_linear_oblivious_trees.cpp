#include "greedy_linear_oblivious_trees.h"

#include <memory>
#include <set>
#include <stdexcept>
#include <chrono>

#include <core/matrix.h>
#include <core/multi_dim_array.h>

#include <models/linear_oblivious_tree.h>

#include <eigen3/Eigen/Core>


#define TIME_BLOCK_START(name) \
    auto begin##name = std::chrono::steady_clock::now(); \
    std::cout << "Starting " << #name << std::endl;

#define TIME_BLOCK_END(name) \
    do { \
        auto end##name = std::chrono::steady_clock::now(); \
        auto time_ms##name = std::chrono::duration_cast<std::chrono::milliseconds>(end##name - begin##name).count(); \
        std::cout << #name << " done in " << time_ms##name << " [ms]" << std::endl; \
    } while (false);


class LinearObliviousTreeLeafLearner : std::enable_shared_from_this<LinearObliviousTreeLeafLearner> {
public:
    LinearObliviousTreeLeafLearner(
            GridPtr grid,
            int nUsedFeatures)
            : grid_(std::move(grid))
            , nUsedFeatures_(nUsedFeatures)
            , stats_(MultiDimArray<1, LinearL2Stat>({(int)grid_->totalBins()}, nUsedFeatures + 1, nUsedFeatures)) {
        id_ = 0;
    }

    double splitScore(const StatBasedLoss<LinearL2Stat>& target, int fId, int condId) {
        int bin = grid_->binOffsets()[fId] + condId;
        int lastBin = (int)grid_->binOffsets()[fId] + (int)grid_->conditionsCount(fId);

        auto leftStat = stats_[bin];
        auto rightStat = stats_[lastBin] - leftStat;

        return target.score(leftStat) + target.score(rightStat);
    }

    void fit(float l2reg) {
        w_ = stats_[grid_->totalBins() - 1].getWHat(l2reg);
    }

    double value(const ConstVecRef<float>& x) const {
        float res = 0.0;

        int i = 0;
        for (auto f : usedFeaturesInOrder_) {
            res += x[f] * (float)w_(i, 0);
            ++i;
        }

        return res;
    }

    std::pair<std::shared_ptr<LinearObliviousTreeLeafLearner>, std::shared_ptr<LinearObliviousTreeLeafLearner>>
    split(int32_t fId, int32_t condId) {
        int origFId = grid_->origFeatureIndex(fId);
        unsigned int nUsedFeatures = nUsedFeatures_ + (1 - usedFeatures_.count(origFId));

        auto left = std::make_shared<LinearObliviousTreeLeafLearner>(grid_, nUsedFeatures);
        auto right = std::make_shared<LinearObliviousTreeLeafLearner>(grid_, nUsedFeatures);

        initChildren(left, right, fId, condId);

        return std::make_pair(left, right);
    }

    void printInfo() {
        std::cout << "LEAF ID: " << id_ << " {" << std::endl;
        printHists();
        printSplits();
        std::cout << "}\n" << std::endl;
    }

    void printHists() {
        for (int fId = 0; fId < grid_->nzFeaturesCount(); ++fId) {
            int offset = grid_->binOffsets()[fId];
            for (int bin = 0; bin <= (int)grid_->conditionsCount(fId); ++bin) {
                int absBin = offset + bin;
                std::cout << "  fId=" << fId << ", bin=" << bin << std::endl;
                std::cout << "    XTX=" << stats_[absBin].getXTX() << ", XTy=" << stats_[absBin].getXTy() << std::endl;
            }
        }
    }

    void printSplits() {
//        for (auto& s : splits_) {
//            auto fId = std::get<0>(s);
//            auto origFId = grid_->origFeatureIndex(fId);
//            auto condId = std::get<1>(s);
//            double minCondition = grid_->condition(fId, 0);
//            double maxCondition = grid_->condition(fId, grid_->conditionsCount(fId) - 1);
//            double condition = grid_->condition(fId, condId);
//            std::cout << "split: fId=" << fId << "(" << origFId << ") " << ", condId=" << condId
//                      << std::setprecision(5) << ", used cond=" << condition
//                      << ", min cond=" << minCondition << ", max cond=" << maxCondition << std::endl;
//        }
    }

private:
    void initChildren(std::shared_ptr<LinearObliviousTreeLeafLearner>& left,
                      std::shared_ptr<LinearObliviousTreeLeafLearner>& right,
                      int32_t splitFId, int32_t condId) {
        left->id_ = 2 * id_;
        right->id_ = 2 * id_ + 1;

        left->usedFeatures_ = usedFeatures_;
        right->usedFeatures_ = usedFeatures_;
        left->usedFeaturesInOrder_ = usedFeaturesInOrder_;
        right->usedFeaturesInOrder_ = usedFeaturesInOrder_;

        int32_t origFeatureId = grid_->origFeatureIndex(splitFId);

        if (usedFeatures_.count(origFeatureId) == 0) {
            left->usedFeatures_.insert(origFeatureId);
            right->usedFeatures_.insert(origFeatureId);
            left->usedFeaturesInOrder_.push_back(origFeatureId);
            right->usedFeaturesInOrder_.push_back(origFeatureId);
        }
    }

private:
    friend class GreedyLinearObliviousTreeLearner;

    GridPtr grid_;
    std::set<int32_t> usedFeatures_;
    std::vector<int32_t> usedFeaturesInOrder_;
    LinearL2Stat::EMx w_;
    MultiDimArray<1, LinearL2Stat> stats_;

    unsigned int nUsedFeatures_;

    int32_t id_;
};


ModelPtr GreedyLinearObliviousTreeLearner::fit(const DataSet& ds, const Target& target) {
    auto beginAll = std::chrono::steady_clock::now();

    auto tree = std::make_shared<LinearObliviousTree>(grid_);

    cacheDs(ds);
    resetState();

    auto bds = cachedBinarize(ds, grid_, fCount_);

    auto ysVec = target.targets();
    auto ys = ysVec.arrayRef();

    auto wsVec = target.weights();
    auto ws = wsVec.arrayRef();

    if (biasCol_ == -1) {
        // TODO
        throw std::runtime_error("provide bias col!");
    }

    std::cout << "start fit" << std::endl;

    TIME_BLOCK_START(BUILDING_ROOT)
    buildRoot(bds, ds, ys, ws);
    TIME_BLOCK_END(BUILDING_ROOT)

    // Root is built

    for (unsigned int d = 0; d < maxDepth_; ++d) {
        TIME_BLOCK_START(UPDATE_NEW_CORRELATIONS)
        updateNewCorrelations(bds, ds, ys, ws);
        TIME_BLOCK_END(UPDATE_NEW_CORRELATIONS)

//        for (auto& l : leaves_) {
//            l->printInfo();
//        }

        TIME_BLOCK_START(FIND_BEST_SPLIT)
        auto split = findBestSplit(target);
        int32_t splitFId = split.first;
        int32_t splitCond = split.second;
        tree->splits_.emplace_back(std::make_tuple(splitFId, splitCond));
        splits_.insert(std::make_pair(splitFId, splitCond));

        int oldNUsedFeatures = usedFeatures_.size();

        int32_t splitOrigFId = grid_->origFeatureIndex(splitFId);
        if (usedFeatures_.count(splitOrigFId) == 0) {
            usedFeatures_.insert(splitOrigFId);
            usedFeaturesOrdered_.push_back(splitOrigFId);
        }
        TIME_BLOCK_END(FIND_BEST_SPLIT)

        TIME_BLOCK_START(INIT_NEW_LEAVES)
        initNewLeaves(split);
        TIME_BLOCK_END(INIT_NEW_LEAVES)

        TIME_BLOCK_START(UPDATE_NEW_LEAVES)
        updateNewLeaves(bds, ds, oldNUsedFeatures, ys, ws);
        TIME_BLOCK_END(UPDATE_NEW_LEAVES)

        leaves_ = newLeaves_;
        newLeaves_.clear();
    }

    TIME_BLOCK_START(FINAL_FIT)
    parallelFor(0, leaves_.size(), [&](int lId) {
        auto& l = leaves_[lId];
        l->fit((float)l2reg_);
    });
    TIME_BLOCK_END(FINAL_FIT)

    std::vector<LinearObliviousTreeLeaf> inferenceLeaves;
    for (auto& l : leaves_) {
        inferenceLeaves.emplace_back(usedFeaturesOrdered_, l->w_);
    }

    tree->leaves_ = std::move(inferenceLeaves);
    return tree;
}

void GreedyLinearObliviousTreeLearner::cacheDs(const DataSet &ds) {
    if (isDsCached_) {
        return;
    }

    for (int fId = 0; fId < (int)grid_->nzFeaturesCount(); ++fId) {
        fColumns_.emplace_back(ds.samplesCount());
        fColumnsRefs_.emplace_back(NULL);
    }

    parallelFor<0>(0, grid_->nzFeaturesCount(), [&](int fId) {
        int origFId = grid_->origFeatureIndex(fId);
        ds.copyColumn(origFId, &fColumns_[fId]);
        fColumnsRefs_[fId] = fColumns_[fId].arrayRef();
    });

    totalBins_ = grid_->totalBins();
    fCount_ = grid_->nzFeaturesCount();
    totalCond_ = totalBins_ - fCount_;
    binOffsets_ = grid_->binOffsets();
    nThreads_ = (int)GlobalThreadPool<0>().numThreads();
    nSamples_ = ds.samplesCount();

    absBinToFId_.resize(totalBins_);
    for (int fId = 0; fId < grid_->nzFeaturesCount(); ++fId) {
        int offset = binOffsets_[fId];
        for (int bin = 0; bin <= grid_->conditionsCount(fId); ++bin) {
            absBinToFId_[offset + bin] = fId;
        }
    }

    fullUpdate_.resize(1U << (unsigned)maxDepth_, false);
    samplesLeavesCnt_.resize(1U << (unsigned)maxDepth_, 0);

    isDsCached_ = true;
}

void GreedyLinearObliviousTreeLearner::resetState() {
    usedFeatures_.clear();
    usedFeaturesOrdered_.clear();
    leafId_.clear();
    leafId_.resize(nSamples_, 0);
    leaves_.clear();
    newLeaves_.clear();
    splits_.clear();
}

void GreedyLinearObliviousTreeLearner::buildRoot(
        const BinarizedDataSet &bds,
        const DataSet &ds,
        ConstVecRef<float> ys,
        ConstVecRef<float> ws) {
    auto root = std::make_shared<LinearObliviousTreeLeafLearner>(this->grid_, 1);
    root->usedFeatures_.insert(biasCol_);
    root->usedFeaturesInOrder_.push_back(biasCol_);

    usedFeatures_.insert(biasCol_);
    usedFeaturesOrdered_.push_back(biasCol_);

    LinearL2StatOpParams params;
    params.vecAddMode = LinearL2StatOpParams::FullCorrelation;

    MultiDimArray<2, LinearL2Stat> stats = ComputeStats<LinearL2Stat>(
            1, leafId_, ds, bds,
            LinearL2Stat(2, 1),
            [&](LinearL2Stat& stat, std::vector<float>& x, int sampleId, int fId) {
        stat.append(x.data(), ys[sampleId], ws[sampleId], params);
    });

    root->stats_ = stats[0].copy();

    leaves_.emplace_back(std::move(root));
}

void GreedyLinearObliviousTreeLearner::updateNewCorrelations(
        const BinarizedDataSet& bds,
        const DataSet& ds,
        ConstVecRef<float> ys,
        ConstVecRef<float> ws) {
    int nUsedFeatures = usedFeaturesOrdered_.size();

    MultiDimArray<2, LinearL2CorStat> stats = ComputeStats<LinearL2CorStat>(
            leaves_.size(), leafId_, ds, bds,
            LinearL2CorStat(nUsedFeatures + 1),
            [&](LinearL2CorStat& stat, std::vector<float>& x, int sampleId, int fId) {
        int origFId = grid_->origFeatureIndex(fId);
        if (usedFeatures_.count(origFId)) return;

        LinearL2CorStatOpParams params;
        params.fVal = ds.fVal(sampleId, origFId);

        stat.append(x.data(), ys[sampleId], ws[sampleId], params);
    });

    // update stats with this correlations
    parallelFor(0, totalBins_, [&](int bin) {
        int origFId = absBinToFId_[bin];
        if (usedFeatures_.count(origFId)) return;

        LinearL2StatOpParams params = {};
        for (int lId = 0; lId < leaves_.size(); ++lId) {
            leaves_[lId]->stats_[bin].append(stats[lId][bin].xxt.data(),
                                             stats[lId][bin].xy, /*unused*/1.0, params);
        }
    });
}

GreedyLinearObliviousTreeLearner::TSplit GreedyLinearObliviousTreeLearner::findBestSplit(
        const Target& target) {
    float bestScore = 1e9;
    int32_t splitFId = -1;
    int32_t splitCond = -1;

    const auto& linearL2Target = dynamic_cast<const LinearL2&>(target);

    MultiDimArray<2, float> splitScores({fCount_, totalCond_});

    // TODO can parallelize by totalBins
    parallelFor(0, fCount_, [&](int fId) {
        for (int cond = 0; cond < grid_->conditionsCount(fId); ++cond) {
            for (auto &l : leaves_) {
                splitScores[fId][cond] += l->splitScore(linearL2Target, fId, cond);
            }
        }
    });

    for (int fId = 0; fId < fCount_; ++fId) {
        for (int cond = 0; cond < grid_->conditionsCount(fId); ++cond) {
            float sScore = splitScores[fId][cond];
            if (sScore < bestScore && splits_.count(std::make_pair(fId, cond)) == 0) {
                bestScore = sScore;
                splitFId = fId;
                splitCond = cond;
            }
        }
    }

    if (splitFId < 0 || splitCond < 0) {
        throw std::runtime_error("Failed to find the best split");
    }

    std::cout << "best split: " << splitFId << " " << splitCond <<  std::endl;

    return std::make_pair(splitFId, splitCond);
}

void GreedyLinearObliviousTreeLearner::initNewLeaves(GreedyLinearObliviousTreeLearner::TSplit split) {
    newLeaves_.clear();

    for (auto& l : leaves_) {
        auto newLeavesPair = l->split(split.first, split.second);
        newLeaves_.emplace_back(newLeavesPair.first);
        newLeaves_.emplace_back(newLeavesPair.second);
    }

    int32_t splitFId = split.first;
    int32_t splitCond = split.second;

    float border = grid_->borders(splitFId).at(splitCond);
    auto fColumnRef = fColumnsRefs_[splitFId];

    for (int i = 0; i < (int)leaves_.size(); ++i) {
        samplesLeavesCnt_[2 * i] = 0;
        samplesLeavesCnt_[2 * i + 1] = 0;
    }

    parallelFor(0,nSamples_, [&](int i) {
        if (fColumnRef[i] <= border) {
            leafId_[i] = 2 * leafId_[i];
        } else {
            leafId_[i] = 2 * leafId_[i] + 1;
        }
        ++samplesLeavesCnt_[leafId_[i]];
    });

    for (int i = 0; i < leaves_.size(); ++i) {
        fullUpdate_[2 * i] = samplesLeavesCnt_[2 * i] <= samplesLeavesCnt_[2 * i + 1];
        fullUpdate_[2 * i + 1] = !fullUpdate_[2 * i];
    }
}

void GreedyLinearObliviousTreeLearner::updateNewLeaves(
        const BinarizedDataSet& bds,
        const DataSet& ds,
        int oldNUsedFeatures,
        ConstVecRef<float> ys,
        ConstVecRef<float> ws) {
    int nUsedFeatures = usedFeaturesOrdered_.size();

    std::vector<int> fullLeafIds(nSamples_, 0);
    std::vector<int> partialLeafIds(nSamples_, 0);
    for (int i = 0; i < nSamples_; ++i) {
        int lId = leafId_[i];
        if (fullUpdate_[lId]) {
            partialLeafIds[i] = -1;
            fullLeafIds[i] = lId / 2;
        } else {
            partialLeafIds[i] = lId / 2;
            fullLeafIds[i] = -1;
        }
    }

    // full updates
    TIME_BLOCK_START(FullUpdatesCompute)
    MultiDimArray<2, LinearL2Stat> fullStats = ComputeStats<LinearL2Stat>(
            leaves_.size(), fullLeafIds, ds, bds,
            LinearL2Stat(nUsedFeatures + 1, nUsedFeatures),
            [&](LinearL2Stat& stat, std::vector<float>& x, int sampleId, int fId) {
        LinearL2StatOpParams params;
        params.vecAddMode = LinearL2StatOpParams::FullCorrelation;
        stat.append(x.data(), ys[sampleId], ws[sampleId], params);
    });
    TIME_BLOCK_END(FullUpdatesCompute)

    TIME_BLOCK_START(FullUpdatesAssign)
    parallelFor(0, newLeaves_.size(), [&](int lId) {
        if (fullUpdate_[lId]) {
            newLeaves_[lId]->stats_ = fullStats[lId / 2].copy();
        }
    });
    TIME_BLOCK_END(FullUpdatesAssign)

    if (oldNUsedFeatures != usedFeatures_.size()) {
        // partial updates
        TIME_BLOCK_START(PartialUpdatesCompute)
        MultiDimArray<2, LinearL2CorStat> partialStats = ComputeStats<LinearL2CorStat>(
                leaves_.size(), partialLeafIds, ds, bds,
                LinearL2CorStat(nUsedFeatures),
                [&](LinearL2CorStat &stat, std::vector<float> &x, int sampleId, int fId) {
                    LinearL2CorStatOpParams params;
                    params.fVal = x[nUsedFeatures - 1];
                    stat.append(x.data(), ys[sampleId], ws[sampleId], params);
                });
        TIME_BLOCK_END(PartialUpdatesCompute)

        TIME_BLOCK_START(PartialUpdatesAssign)
        LinearL2StatOpParams params;
        params.shift = -1;

        parallelFor(0, newLeaves_.size(), [&](int lId) {
            if (!fullUpdate_[lId]) {
                for (int bin = 0; bin < totalBins_; ++bin) {
                    auto &partialStat = partialStats[lId / 2][bin];
                    newLeaves_[lId]->stats_[bin].append(partialStat.xxt.data(),
                                                        partialStat.xy, /*unused*/1.0, params);
                }
            }
        });
        TIME_BLOCK_END(PartialUpdatesAssign)
    }

    TIME_BLOCK_START(SubtractLeftsFromParents)
    // subtract lefts from parents to obtain inner parts of right children

    parallelFor(0, leaves_.size(), [&](int lId) {
        auto& parent = leaves_[lId];
        auto& left = newLeaves_[2 * lId];
        auto& right = newLeaves_[2 * lId + 1];

        // This - and += ops will only update inner correlations -- exactly what we need
        // new feature correlation will stay the same

        if (fullUpdate_[left->id_]) {
            for (int bin = 0; bin < totalBins_; ++bin) {
                right->stats_[bin] += parent->stats_[bin] - left->stats_[bin];
            }
        } else {
            for (int bin = 0; bin < totalBins_; ++bin) {
                left->stats_[bin] += parent->stats_[bin] - right->stats_[bin];
            }
        }
    });

    TIME_BLOCK_END(SubtractLeftsFromParents)
}
