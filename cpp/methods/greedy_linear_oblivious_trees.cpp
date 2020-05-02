#include "greedy_linear_oblivious_trees.h"

#include <memory>
#include <set>
#include <stdexcept>
#include <chrono>
#include <functional>

#include <core/matrix.h>
#include <core/multi_dim_array.h>

#include <models/linear_oblivious_tree.h>

#include <eigen3/Eigen/Core>


#define TIME_BLOCK_START(name)
#define TIME_BLOCK_END(name)

//#define TIME_BLOCK_START(name) \
//    auto begin##name = std::chrono::steady_clock::now(); \
//    std::cout << "Starting " << #name << std::endl;
//
//#define TIME_BLOCK_END(name) \
//    do { \
//        auto end##name = std::chrono::steady_clock::now(); \
//        auto time_ms##name = std::chrono::duration_cast<std::chrono::milliseconds>(end##name - begin##name).count(); \
//        std::cout << #name << " done in " << time_ms##name << " [ms]" << std::endl; \
//    } while (false);

GreedyLinearObliviousTreeLearnerOptions GreedyLinearObliviousTreeLearnerOptions::fromJson(const json& params) {
    GreedyLinearObliviousTreeLearnerOptions opts;
    opts.l2reg = params["l2reg"];
    opts.maxDepth = params["depth"];
    return opts;
}


class LinearObliviousTreeLeafLearner : std::enable_shared_from_this<LinearObliviousTreeLeafLearner> {
public:
    LinearObliviousTreeLeafLearner(
            GridPtr grid,
            int nUsedFeatures)
            : grid_(std::move(grid))
            , stats_(MultiDimArray<1, LinearL2Stat>({(int)grid_->totalBins()}, nUsedFeatures + 1, nUsedFeatures))
            , nUsedFeatures_(nUsedFeatures) {
        id_ = 0;
        (void)nUsedFeatures_;
    }

    double splitScore(const StatBasedLoss<LinearL2Stat>& target, int fId, int condId) {
        int bin = grid_->binOffsets()[fId] + condId;
        int lastBin = (int)grid_->binOffsets()[fId] + (int)grid_->conditionsCount(fId);

        auto& leftStat = stats_[bin];
        auto rightStat = stats_[lastBin] - leftStat;

        return target.score(leftStat) + target.score(rightStat);
    }

    void fit(double l2reg, int size = 0) {
        w_ = stats_[grid_->totalBins() - 1].getWHat(l2reg, size);
    }

    std::pair<std::shared_ptr<LinearObliviousTreeLeafLearner>, std::shared_ptr<LinearObliviousTreeLeafLearner>>
    split(int32_t fId, int32_t condId, int nUsedFeatures) {

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
                if (bin == 0) {
                    std::cout << "    " << stats_[absBin] << std::endl;
                } else {
                    std::cout << "    " << (stats_[absBin] - stats_[absBin - 1]) << std::endl;
                }
                std::cout << std::endl;
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
    }

private:
    friend class GreedyLinearObliviousTreeLearner;

    GridPtr grid_;
//    std::set<int32_t> usedFeatures_;
//    std::vector<int32_t> usedFeaturesInOrder_;
    LinearL2Stat::EMx w_;
    MultiDimArray<1, LinearL2Stat> stats_;

    unsigned int nUsedFeatures_;

    int32_t id_;
};


ModelPtr GreedyLinearObliviousTreeLearner::fit(const DataSet& ds, const Target& target) {
    auto tree = std::make_shared<LinearObliviousTree>(grid_);

    cacheDs(ds);
    resetState();

    auto bds = cachedBinarize(ds, grid_, fCount_);

    auto indices = target.indices();
    indices_ = indices.arrayRef();
    nSamples_ = indices.size();

    auto ysVec = target.targets();
    auto ys = ysVec.arrayRef();

    auto wsVec = target.weights();
    auto ws = wsVec.arrayRef();

    if (opts_.biasCol == -1) {
        // TODO
        throw std::runtime_error("provide bias col!");
    }

    std::cout << "start fit" << std::endl;

    TIME_BLOCK_START(BUILDING_ROOT)
    buildRoot(bds, ds, ys, ws);
    TIME_BLOCK_END(BUILDING_ROOT)

    double currentScore = 1e+9;

    // Root is built

    for (unsigned int d = 0; d < opts_.maxDepth; ++d) {
//        for (int i = 0; i < nSamples_; ++i) {
//            std::cout << i << " goes to " << leafId_[i] << std::endl;
//        }

        TIME_BLOCK_START(UPDATE_NEW_CORRELATIONS)
        updateNewCorrelations(bds, ds, ys, ws);
        TIME_BLOCK_END(UPDATE_NEW_CORRELATIONS)

//        for (auto& l : leaves_) {
//            l->printInfo();
//        }

        TIME_BLOCK_START(FIND_BEST_SPLIT)
        auto split = findBestSplit(target);

        double splitScore = std::get<0>(split);
        if (splitScore >= currentScore - 1e-9) {
            break;
        }

        currentScore = splitScore;
        int32_t splitFId = std::get<1>(split);
        int32_t splitCond = std::get<2>(split);
        tree->splits_.emplace_back(std::make_tuple(splitFId, splitCond));
        splits_.insert(split);

        int oldNUsedFeatures = (int)usedFeatures_.size();

        int32_t splitOrigFId = grid_->origFeatureIndex(splitFId);
        if (usedFeatures_.count(splitOrigFId) == 0) {
            usedFeatures_.insert(splitOrigFId);
            usedFeaturesOrdered_.push_back(splitOrigFId);
            updateXs(splitOrigFId);
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
        l->fit(opts_.l2reg, usedFeatures_.size());
    });
    TIME_BLOCK_END(FINAL_FIT)

    std::vector<LinearObliviousTreeLeaf> inferenceLeaves;
    for (auto& l : leaves_) {
        inferenceLeaves.emplace_back(usedFeaturesOrdered_, l->w_, l->stats_[totalBins_ - 1].w_);
    }

    tree->leaves_ = std::move(inferenceLeaves);

    std::cout << "Resulting tree:" << std::endl;
    tree->printInfo();

    return tree;
}

void GreedyLinearObliviousTreeLearner::cacheDs(const DataSet &ds) {
    if (isDsCached_) {
        return;
    }

    for (int fId = 0; fId < ds.featuresCount(); ++fId) {
        fColumns_.emplace_back(ds.samplesCount());
        fColumnsRefs_.emplace_back(NULL);
    }

    parallelFor(0, ds.featuresCount(), [&](int fId) {
        ds.copyColumn(fId, &fColumns_[fId]);
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

    fullUpdate_.resize(1U << (unsigned)opts_.maxDepth, false);
    samplesLeavesCnt_.resize(1U << (unsigned)opts_.maxDepth, 0);

    auto sizeV = std::vector<int>({nThreads_});
    corStats_ = std::make_unique<MultiDimArray<1, MultiDimArray<2, LinearL2CorStat>>>(sizeV);
    stats_ = std::make_unique<MultiDimArray<1, MultiDimArray<2, LinearL2Stat>>>(sizeV);

    LinearL2CorStat defaultCorStat(opts_.maxDepth + 1, 0);
    LinearL2Stat defaultStat(opts_.maxDepth + 1, 0);

    parallelFor(0, nThreads_, [&](int thId) {
        (*corStats_)[thId] = MultiDimArray<2, LinearL2CorStat>({1 << opts_.maxDepth, totalBins_}, defaultCorStat);
        (*stats_)[thId] = MultiDimArray<2, LinearL2Stat>({1 << opts_.maxDepth, totalBins_}, defaultStat);
    });

    xs_ = MultiDimArray<2, float>({(int)ds.samplesCount(), opts_.maxDepth + 1});

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

void GreedyLinearObliviousTreeLearner::resetStats(int nLeaves, int filledSize) {
    parallelFor(0, nThreads_, [&](int thId) {
        auto& thStats = (*stats_)[thId];
        auto& thCorStats = (*corStats_)[thId];
        for (int lId = 0; lId < nLeaves; ++lId) {
            auto leafCorStat = thCorStats[lId];
            auto leafStat = thStats[lId];
            for (int bin = 0; bin < totalBins_; ++bin) {
                auto& corStat = leafCorStat[bin];
                corStat.reset();
                corStat.setFilledSize(filledSize);
                auto& stat = leafStat[bin];
                stat.reset();
                stat.setFilledSize(filledSize);
            }
        }
    });
}

void GreedyLinearObliviousTreeLearner::updateXs(int origFId) {
    int pos = usedFeatures_.size() - 1;
    auto fColumn = fColumnsRefs_[origFId];
    parallelFor(0, nSamples_, [&](int i) {
        xs_[i][pos] = fColumn[indices_[i]];
    });
}

float* GreedyLinearObliviousTreeLearner::curX(int sampleId) {
    return xs_[sampleId].data();
}

void GreedyLinearObliviousTreeLearner::buildRoot(
        const BinarizedDataSet &bds,
        const DataSet &ds,
        ConstVecRef<float> ys,
        ConstVecRef<float> ws) {
    auto root = std::make_shared<LinearObliviousTreeLeafLearner>(this->grid_, 1);
    usedFeatures_.insert(opts_.biasCol);
    usedFeaturesOrdered_.push_back(opts_.biasCol);
    updateXs(opts_.biasCol);

    LinearL2StatOpParams params;
    params.vecAddMode = LinearL2StatOpParams::FullCorrelation;

    resetStats(1, 1);

    ComputeStats<LinearL2Stat>(
            1, leafId_, ds, bds,
            *stats_,
            [&](LinearL2Stat& stat, int sampleId, int origFId) {
        float* x = curX(sampleId);
        stat.append(x, ys[sampleId], ws[sampleId], params);
    });

    auto stats = (*stats_)[0][0];

    parallelFor(0, totalBins_, [&](int bin) {
        root->stats_[bin] = stats[bin];
    });

    leaves_.emplace_back(std::move(root));
}

void GreedyLinearObliviousTreeLearner::updateNewCorrelations(
        const BinarizedDataSet& bds,
        const DataSet& ds,
        ConstVecRef<float> ys,
        ConstVecRef<float> ws) {
    int nUsedFeatures = usedFeaturesOrdered_.size();
    resetStats(leaves_.size(), nUsedFeatures + 1);

    ComputeStats<LinearL2CorStat>(
            leaves_.size(), leafId_, ds, bds,
            *corStats_,
            [&](LinearL2CorStat& stat, int sampleId, int origFId) {
        if (usedFeatures_.count(origFId)) return;

        LinearL2CorStatOpParams params;
        params.fVal = fColumnsRefs_[origFId][indices_[sampleId]];

        float* x = curX(sampleId);

        stat.append(x, ys[sampleId], ws[sampleId], params);
    });

    MultiDimArray<2, LinearL2CorStat>& stats = (*corStats_)[0];

    // update stats with this correlations
    parallelFor(0, totalBins_, [&](int bin) {
        int fId = absBinToFId_[bin];
        int origFId = grid_->origFeatureIndex(fId);
        if (usedFeatures_.count(origFId)) return;

        LinearL2StatOpParams params = {};
        for (uint64_t lId = 0; lId < leaves_.size(); ++lId) {
            auto& stat = stats[lId][bin];
            leaves_[lId]->stats_[bin].append(stat.xxt.data(),
                                             stat.xy,
                                             stat.sumX, params);
        }
    });
}

GreedyLinearObliviousTreeLearner::TSplit GreedyLinearObliviousTreeLearner::findBestSplit(
        const Target& target) {
    double bestScore = 1e9;
    int32_t splitFId = -1;
    int32_t splitCond = -1;

    const auto& linearL2Target = dynamic_cast<const LinearL2&>(target);

    MultiDimArray<2, double> splitScores({fCount_, totalCond_});

    // TODO can parallelize by totalBins
//    parallelFor(0, fCount_, [&](int fId) {
//        for (int cond = 0; cond < grid_->conditionsCount(fId); ++cond) {
//            for (auto &l : leaves_) {
//                splitScores[fId][cond] += l->splitScore(linearL2Target, fId, cond);
//            }
//        }
//    });

    parallelFor(0, totalBins_, [&](int bin) {
        int fId = absBinToFId_[bin];
        int cond = bin - binOffsets_[fId];
        if (cond != grid_->conditionsCount(fId)) {
            for (auto &l : leaves_) {
                splitScores[fId][cond] += l->splitScore(linearL2Target, fId, cond);
            }
        }
    });

    for (int fId = 0; fId < fCount_; ++fId) {
        for (int cond = 0; cond < grid_->conditionsCount(fId); ++cond) {
            double sScore = splitScores[fId][cond];
//            double border = grid_->borders(fId)[cond];
//            std::cout << "split score fId=" << fId << ", cond=" << cond << " (" << border << "): " << sScore << std::endl;
            if (sScore < bestScore) {
                bestScore = sScore;
                splitFId = fId;
                splitCond = cond;
            }
        }
    }

    if (splitFId < 0 || splitCond < 0) {
        throw std::runtime_error("Failed to find the best split");
    }

//    std::cout << "best split: " << splitFId << " " << splitCond <<  std::endl;

    return std::make_tuple(bestScore, splitFId, splitCond);
}

void GreedyLinearObliviousTreeLearner::initNewLeaves(GreedyLinearObliviousTreeLearner::TSplit split) {
    newLeaves_.clear();

    auto splitFId = std::get<1>(split);
    int splitOrigFId = grid_->origFeatureIndex(splitFId);
    auto splitCond = std::get<2>(split);

    newLeaves_.resize(2 * leaves_.size());

    parallelFor(0, leaves_.size(), [&](int lId) {
        auto newLeavesPair = leaves_[lId]->split(splitFId, splitCond, usedFeatures_.size());
        newLeaves_[2 * lId] = newLeavesPair.first;
        newLeaves_[2 * lId + 1] = newLeavesPair.second;
    });

    auto border = grid_->borders(splitFId).at(splitCond);
    auto fColumnRef = fColumnsRefs_[splitOrigFId];

    for (int i = 0; i < (int)leaves_.size(); ++i) {
        samplesLeavesCnt_[2 * i] = 0;
        samplesLeavesCnt_[2 * i + 1] = 0;
    }

    parallelFor(0,nSamples_, [&](int i) {
        if (fColumnRef[indices_[i]] <= border) {
            leafId_[i] = 2 * leafId_[i];
        } else {
            leafId_[i] = 2 * leafId_[i] + 1;
        }
        ++samplesLeavesCnt_[leafId_[i]];
    });

    for (uint64_t i = 0; i < leaves_.size(); ++i) {
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

    resetStats(leaves_.size(), nUsedFeatures);

    // full updates
    TIME_BLOCK_START(FullUpdatesCompute)
    ComputeStats<LinearL2Stat>(
            leaves_.size(), fullLeafIds, ds, bds,
            *stats_,
            [&](LinearL2Stat& stat, int sampleId, int origFId) {
        LinearL2StatOpParams params;
        params.vecAddMode = LinearL2StatOpParams::FullCorrelation;
        float* x = curX(sampleId);
        stat.append(x, ys[sampleId], ws[sampleId], params);
    });
    TIME_BLOCK_END(FullUpdatesCompute)

    auto& fullStats = (*stats_)[0];

    TIME_BLOCK_START(FullUpdatesAssign)
//    parallelFor(0, newLeaves_.size(), [&](int lId) {
//        if (fullUpdate_[lId]) {
//            newLeaves_[lId]->stats_ = fullStats[lId / 2].copy();
//        }
//    });

    parallelFor(0, totalBins_, [&](int bin) {
        for (int lId = 0; lId < (int)newLeaves_.size(); ++lId) {
            if (!fullUpdate_[lId]) continue;
//            auto &fullStat = fullStats[lId / 2][bin];
            newLeaves_[lId]->stats_[bin] = fullStats[lId / 2][bin];
        }
    });
    TIME_BLOCK_END(FullUpdatesAssign)

    if (oldNUsedFeatures != usedFeatures_.size()) {
        // partial updates

        // corStats have already been reset

        TIME_BLOCK_START(PartialUpdatesCompute)
        ComputeStats<LinearL2CorStat>(
                leaves_.size(), partialLeafIds, ds, bds,
                *corStats_,
                [&](LinearL2CorStat& stat, int sampleId, int origFId) {
                    float* x = curX(sampleId);
                    LinearL2CorStatOpParams params;
                    params.fVal = x[nUsedFeatures - 1];
                    stat.append(x, ys[sampleId], ws[sampleId], params);
                });
        TIME_BLOCK_END(PartialUpdatesCompute)

        auto& partialStats = (*corStats_)[0];

        TIME_BLOCK_START(PartialUpdatesAssign)
        LinearL2StatOpParams params;
        params.shift = -1;

//        parallelFor(0, newLeaves_.size(), [&](int lId) {
//            if (fullUpdate_[lId]) return;
//            for (int bin = 0; bin < totalBins_; ++bin) {
//                auto &partialStat = partialStats[lId / 2][bin];
//                newLeaves_[lId]->stats_[bin].append(partialStat.xxt.data(),
//                                                    partialStat.xy,
//                                                    partialStat.sumX, params);
//            }
//        });

        parallelFor(0, totalBins_, [&](int bin) {
            for (int lId = 0; lId < (int)newLeaves_.size(); ++lId) {
                if (fullUpdate_[lId]) continue;
                auto &partialStat = partialStats[lId / 2][bin];
                newLeaves_[lId]->stats_[bin].append(partialStat.xxt.data(),
                                                    partialStat.xy,
                                                    partialStat.sumX, params);
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

        LinearL2StatOpParams params;
        params.opSize = oldNUsedFeatures;

        if (fullUpdate_[left->id_]) {
            for (int bin = 0; bin < totalBins_; ++bin) {
                right->stats_[bin].append(parent->stats_[bin] - left->stats_[bin], params);
            }
        } else {
            for (int bin = 0; bin < totalBins_; ++bin) {
                left->stats_[bin].append(parent->stats_[bin] - right->stats_[bin], params);
            }
        }
    });

    TIME_BLOCK_END(SubtractLeftsFromParents)
}
