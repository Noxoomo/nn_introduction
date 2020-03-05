#include "greedy_linear_oblivious_trees_v2.h"

#include <memory>
#include <set>
#include <stdexcept>
#include <chrono>
#include <cstring>

#include <core/vec_factory.h>
#include <core/matrix.h>

#include <eigen3/Eigen/Core>


HistogramV2::HistogramV2(BinarizedDataSet& bds, GridPtr grid, unsigned int nUsedFeatures, int lastUsedFeatureId/*,
        BinStat* stats*/)
        : bds_(bds)
        , grid_(std::move(grid))
        , nUsedFeatures_(nUsedFeatures)
        , lastUsedFeatureId_(lastUsedFeatureId) {
    hist_ = std::vector<BinStat>(grid_->totalBins(), BinStat(nUsedFeatures_ + 1, nUsedFeatures));
}

void HistogramV2::addNewCorrelation(int bin, const float* xtx, float xty, int shift) {
    hist_[bin].addNewCorrelation(xtx, xty, shift);
}

void HistogramV2::addBinStat(int bin, const BinStat& stats) {
    hist_[bin] += stats;
}

void HistogramV2::prefixSumBins() {
    parallelFor<1>(0, (int)grid_->nzFeaturesCount(), [&](int fId) {
        int offset = grid_->binOffsets()[fId];
        for (int localBinId = 1; localBinId <= grid_->conditionsCount(fId); ++localBinId) {
            int bin = offset + localBinId;
            hist_[bin] += hist_[bin - 1];
        }
    });
}

std::shared_ptr<BinStat::EMx> HistogramV2::getW(double l2reg) {
    if (lastUsedFeatureId_ == -1) {
        throw std::runtime_error("No features are used");
    }

    uint32_t offset = grid_->binOffsets().at(lastUsedFeatureId_);
    uint32_t lastPos = offset + grid_->conditionsCount(lastUsedFeatureId_);

    return std::make_shared<BinStat::EMx>(hist_[lastPos].getW(l2reg));
}

std::pair<double, double> HistogramV2::splitScore(int fId, int condId, double l2reg,
                                                double traceReg) {
    uint32_t offset = grid_->binOffsets()[fId];
    uint32_t binPos = offset + condId;
    uint32_t lastPos = offset + grid_->conditionsCount(fId);

    auto resLeft = BinStat::fitScore(hist_[binPos], l2reg);
    auto resRight = BinStat::fitScore(hist_[lastPos] - hist_[binPos], l2reg);

    return std::make_pair(resLeft, resRight);
}

void HistogramV2::print() {
    std::cout << "Hist (nUsedFeatures=" << nUsedFeatures_ << ") {" << std::endl;
    for (int fId = 0; fId < grid_->nzFeaturesCount(); ++fId) {
        std::cout << "fId: " << fId << std::endl;
        for (int cond = 0; cond <= grid_->conditionsCount(fId); ++cond) {
            uint32_t offset = grid_->binOffsets().at(fId);
            uint32_t bin = offset + cond;
            std::cout << "fId: " << fId << ", cond: " << cond << ", XTX:\n" << hist_[bin].getXTX()
                    << ",\nXTy:\n" << hist_[bin].getXTy()
                    << ", weight: " << hist_[bin].getWeight() << std::endl;
        }
    }
    std::cout << "}" << std::endl;
}

HistogramV2& HistogramV2::operator+=(const HistogramV2& h) {
    for (int bin = 0; bin < (int)grid_->totalBins(); ++bin) {
        hist_[bin] += h.hist_[bin];
    }

    return *this;
}

HistogramV2& HistogramV2::operator-=(const HistogramV2& h) {
    for (int bin = 0; bin < (int)grid_->totalBins(); ++bin) {
        hist_[bin] -= h.hist_[bin];
    }

    return *this;
}


class LinearObliviousTreeLeafV2 : std::enable_shared_from_this<LinearObliviousTreeLeafV2> {
public:
    LinearObliviousTreeLeafV2(BinarizedDataSet& bds, GridPtr grid, double l2reg, double traceReg, unsigned int maxDepth,
            unsigned int nUsedFeatures, int lastUsedFeatureId/*, BinStat* stats*/)
            : bds_(bds)
            , grid_(std::move(grid))
            , l2reg_(l2reg)
            , traceReg_(traceReg)
            , maxDepth_(maxDepth)
            , nUsedFeatures_(nUsedFeatures)
            , lastUsedFeatureId_(lastUsedFeatureId) {
        hist_ = std::make_unique<HistogramV2>(bds_, grid_, nUsedFeatures, lastUsedFeatureId/*, stats*/);
        id_ = 0;
    }

    double splitScore(int fId, int condId) {
        auto sScore = hist_->splitScore(fId, condId, l2reg_, traceReg_);
        return sScore.first + sScore.second;
    }

    void fit() {
        if (w_) {
            return;
        }
        w_ = hist_->getW(l2reg_);
    }

    double value(const ConstVecRef<float>& x) {
        double res = 0.0;

        int i = 0;
        for (auto f : usedFeaturesInOrder_) {
            res += (BinStat::fType)(x[f]) * (*w_)(i++, 0);
        }

        return res;
    }

    std::pair<std::shared_ptr<LinearObliviousTreeLeafV2>, std::shared_ptr<LinearObliviousTreeLeafV2>>
    split(int32_t fId, int32_t condId) {
        int origFId = grid_->origFeatureIndex(fId);
        unsigned int nUsedFeatures = nUsedFeatures_ + (1 - usedFeatures_.count(origFId));
//        std::cout << "new nUsedFeatures: " << nUsedFeatures << std::endl;

//        auto skip = grid_->totalBins();
        auto left = std::make_shared<LinearObliviousTreeLeafV2>(bds_, grid_,
                l2reg_, traceReg_,
                maxDepth_, nUsedFeatures, fId/*, stats + 2 * id_ * skip*/);
        auto right = std::make_shared<LinearObliviousTreeLeafV2>(bds_, grid_,
                l2reg_, traceReg_,
                maxDepth_, nUsedFeatures, fId/*, stats + (2 * id_ + 1) * skip*/);

        initChildren(left, right, fId, condId);

        return std::make_pair(left, right);
    }

    void printHists() {
        hist_->print();
    }

    void printInfo() {
        printSplits();
        std::cout << std::endl;
    }

    void printSplits() {
        for (auto& s : splits_) {
            auto fId = std::get<0>(s);
            auto origFId = grid_->origFeatureIndex(fId);
            auto condId = std::get<1>(s);
            double minCondition = grid_->condition(fId, 0);
            double maxCondition = grid_->condition(fId, grid_->conditionsCount(fId) - 1);
            double condition = grid_->condition(fId, condId);
            std::cout << "split: fId=" << fId << "(" << origFId << ") " << ", condId=" << condId
                      << std::setprecision(5) << ", used cond=" << condition
                      << ", min cond=" << minCondition << ", max cond=" << maxCondition << std::endl;
        }
    }

private:
    void initChildren(std::shared_ptr<LinearObliviousTreeLeafV2>& left,
                      std::shared_ptr<LinearObliviousTreeLeafV2>& right,
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

        left->splits_ = this->splits_;
        left->splits_.emplace_back(std::make_tuple(splitFId, condId, true));
        right->splits_ = this->splits_;
        right->splits_.emplace_back(std::make_tuple(splitFId, condId, false));
    }

private:
    friend class GreedyLinearObliviousTreeLearnerV2;
    friend class LinearObliviousTreeV2;

    BinarizedDataSet& bds_;

    GridPtr grid_;
    std::set<int32_t> usedFeatures_;
    std::vector<int32_t> usedFeaturesInOrder_;
    std::shared_ptr<BinStat::EMx> w_;
    std::vector<std::tuple<int32_t, int32_t, bool>> splits_;

    double l2reg_;
    double traceReg_;

    unsigned int maxDepth_;
    unsigned int nUsedFeatures_;
    int lastUsedFeatureId_;

    int32_t id_;

    std::unique_ptr<HistogramV2> hist_;
};



ModelPtr GreedyLinearObliviousTreeLearnerV2::fit(const DataSet& ds, const Target& target) {
    auto beginAll = std::chrono::steady_clock::now();

    auto tree = std::make_shared<LinearObliviousTreeV2>(grid_);

    cacheDs(ds);
    resetState();

    // todo cache
    auto bds = cachedBinarize(ds, grid_, fCount_);

    auto ysVec = target.targets();
    auto ys = ysVec.arrayRef();

    auto wsVec = target.weights();
    auto ws = wsVec.arrayRef();

    std::cout << "start" << std::endl;

    auto root = initRoot(ds, bds, ys, ws);
//    root->hist_->print();
    leaves_.emplace_back(std::move(root));

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();



    // Root is built





    for (unsigned int d = 0; d < maxDepth_; ++d) {
        // Update new correlations

        begin = std::chrono::steady_clock::now();

        auto nUsedFeatures = leaves_[0]->nUsedFeatures_;
        computeNewCorrelations(ds, bds, ys, ws, nUsedFeatures);

        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "2 in " << time_ms << " [ms]" << std::endl;

//        for (int i = 0; i < leaves.size(); ++i) {
//            std::cout << i << std::endl;
//            leaves[i]->hist_->print();
//        }

        // Find best split





        begin = std::chrono::steady_clock::now();

        auto split = findBestSplit();
        int splitFId = std::get<0>(split);
        int splitCond = std::get<1>(split);

        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "Best split found in " << time_ms << "[ms], splitting" << std::endl;

//        if (std::get<2>(split) < -1000) {
//            for (auto& l : leaves_) {
//                std::cout << "===" << std::endl;
//                l->printHists();
//            }
//            leaves_[0]->printInfo();
//        }

        tree->splits_.emplace_back(std::make_tuple(splitFId, splitCond));

//        std::cout << "splitFId=" << splitFId << ", splitCond=" << splitCond << std::endl;

        // Split





        auto beginSP = std::chrono::steady_clock::now();

        // 1) find new leaf ids

        begin = std::chrono::steady_clock::now();

        float border = grid_->borders(splitFId).at(splitCond);
        auto fColumnRef = fColumnsRefs_[splitFId];

        for (int i = 0; i < leaves_.size(); ++i) {
            samplesLeavesCnt_[2 * i] = 0;
            samplesLeavesCnt_[2 * i + 1] = 0;
        }

        parallelFor(0, ds.samplesCount(), [&](int i) {
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

        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "1) done in" << time_ms << "[ms]" << std::endl;

        // 2) init new leaves

        begin = std::chrono::steady_clock::now();

        std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> newLeaves;
        for (int i = 0; i < (int)leaves_.size() * 2; ++i) {
            newLeaves.emplace_back(std::shared_ptr<LinearObliviousTreeLeafV2>(nullptr));
        }

        parallelFor(0, leaves_.size(), [&](int lId) {
            auto& l = leaves_[lId];
            auto splits = l->split(splitFId, splitCond);
            newLeaves[splits.first->id_] = std::move(splits.first);
            newLeaves[splits.second->id_] = std::move(splits.second);
        });

        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "2) done in" << time_ms << "[ms]" << std::endl;

        // 3) update current ds, reset stats

        begin = std::chrono::steady_clock::now();

        int32_t splitOrigFId = grid_->origFeatureIndex(splitFId);
        if (usedFeatures_.count(splitOrigFId) == 0) {
            usedFeatures_.insert(splitOrigFId);
            usedFeaturesOrdered_.push_back(splitOrigFId);
        }
        auto oldNUsedFeatures = nUsedFeatures;
        nUsedFeatures = usedFeatures_.size();

//        std::cout << 3.4 << std::endl;

        memset(h_XTX_.data_ptr(), 0, h_XTX_.nbytes());
        memset(h_XTy_.data_ptr(), 0, h_XTy_.nbytes());

        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "3.1) done in" << time_ms << "[ms]" << std::endl;

        begin = std::chrono::steady_clock::now();

        parallelFor(0, nThreads_ + 2, [&](int thId) {
            if (thId == curLeavesCoord_) return;
            for (int lId = 0; lId < (int) newLeaves.size(); ++lId) {
                for (int bin = 0; bin < totalBins_; ++bin) {
                    stats_[thId][lId][bin].reset();
                    stats_[thId][lId][bin].setFilledSize(nUsedFeatures);
                }
            }
        });

        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "3.2) done in" << time_ms << "[ms]" << std::endl;

        // 4) build full correlations only for left children, update new correlations for right ones

        begin = std::chrono::steady_clock::now();

        parallelFor(0, nSamples_, [&](int blockId, int i) {
            auto& x = curX_[blockId];
            ds.fillSample(i, usedFeaturesOrdered_, x);
            int lId = leafId_[i];
            auto bins = bds.sampleBins(i); // todo cache
            float y = ys[i];
            float w = ws[i];

            if (fullUpdate_[lId]) {
                for (int fId = 0; fId < fCount_; ++fId) {
                    int bin = (int) binOffsets_[fId] + bins[fId];
                    stats_[2 + blockId][lId][bin].addFullCorrelation(x.data(), y, w);
                }
            } else {
                if (nUsedFeatures > oldNUsedFeatures) {
                    float fVal = x[oldNUsedFeatures];
                    for (int fId = 0; fId < fCount_; ++fId) {
                        int bin = (int) binOffsets_[fId] + bins[fId];

                        for (unsigned int f = 0; f < oldNUsedFeatures; ++f) {
                            h_XTX_ref_[blockId][lId][bin][f] += x[f] * fVal * w;
                        }
                        h_XTX_ref_[blockId][lId][bin][oldNUsedFeatures] += fVal * fVal * w;
                        h_XTy_ref_[blockId][lId][bin] += fVal * y * w;
                    }
                }
            }
        });

        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "4.1) done in" << time_ms << "[ms]" << std::endl;

        begin = std::chrono::steady_clock::now();

        // for right leaves, prefix sum new correlations
        if (nUsedFeatures > oldNUsedFeatures) {
            // todo change order?
            parallelFor(0, fCount_, [&](int fId) {
                for (int localBinId = 0; localBinId <= (int)grid_->conditionsCount(fId); ++localBinId) {
                    int bin = binOffsets_[fId] + localBinId;
                    for (int lId = 0; lId < (int)newLeaves.size(); ++lId) {
                        if (!fullUpdate_[lId]) {
                            for (int thId = 0; thId < nThreads_; ++thId) {
                                if (localBinId != 0) {
                                    for (unsigned int i = 0; i <= oldNUsedFeatures; ++i) {
                                        h_XTX_ref_[thId][lId][bin][i] += h_XTX_ref_[thId][lId][bin - 1][i];
                                    }
                                    h_XTy_ref_[thId][lId][bin] += h_XTy_ref_[thId][lId][bin - 1];
                                }
                                newLeaves[lId]->hist_->addNewCorrelation(bin, h_XTX_ref_[thId][lId][bin].data(),
                                                                          h_XTy_ref_[thId][lId][bin], -1);
                            }
                        }
                    }
                }
            });
        }

        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "4.2) done in" << time_ms << "[ms]" << std::endl;

        begin = std::chrono::steady_clock::now();

        // For left leaves, sum up stats and then compute prefix sums
        parallelFor(0, totalBins_, [&](int bin) {
            for (int lId = 0; lId < (int)newLeaves.size(); ++lId) {
                if (fullUpdate_[lId]) {
                    for (int blockId = 0; blockId < nThreads_; ++blockId) {
                        newLeaves[lId]->hist_->addBinStat(bin, stats_[2 + blockId][lId][bin]);
                    }
                }
            }
        });

        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "4.3) done in" << time_ms << "[ms]" << std::endl;

        begin = std::chrono::steady_clock::now();

        parallelFor(0, newLeaves.size(), [&](int lId) {
            if (fullUpdate_[lId]) {
                newLeaves[lId]->hist_->prefixSumBins();
            }
        });

        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "4.4) done in" << time_ms << "[ms]" << std::endl;

        // subtract lefts from parents to obtain inner parts of right children

        begin = std::chrono::steady_clock::now();

        parallelFor(0, leaves_.size(), [&](int lId) {
            auto& parent = leaves_[lId];
            auto& left = newLeaves[2 * lId];
            auto& right = newLeaves[2 * lId + 1];

            // This - and += ops will only update inner correlations -- exactly what we need
            // new feature correlation will stay the same

            if (fullUpdate_[left->id_]) {
                for (int bin = 0; bin < totalBins_; ++bin) {
                    right->hist_->hist_[bin] += parent->hist_->hist_[bin] - left->hist_->hist_[bin];
                }
            } else {
                for (int bin = 0; bin < totalBins_; ++bin) {
                    left->hist_->hist_[bin] += parent->hist_->hist_[bin] - right->hist_->hist_[bin];
                }
            }
        });

        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "4.5) done in" << time_ms << "[ms]" << std::endl;

//        std::cout << 4.6 << std::endl;

        leaves_ = std::move(newLeaves);
        curLeavesCoord_ ^= 1U;

        auto endSP = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(endSP - beginSP).count();
        std::cout << "Split done in" << time_ms << "[ms]" << std::endl;
    }

    parallelFor(0, leaves_.size(), [&](int lId) {
        auto& l = leaves_[lId];
        l->fit();
    });

    tree->leaves_ = std::move(leaves_);

    auto endAll = std::chrono::steady_clock::now();
    time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(endAll - beginAll).count();
    std::cout << "All fit done in " << time_ms << "[ms]" << std::endl;

    return tree;
}

std::shared_ptr<LinearObliviousTreeLeafV2> GreedyLinearObliviousTreeLearnerV2::initRoot(const DataSet& ds, BinarizedDataSet& bds,
                                                                                        VecRef<float> ys, VecRef<float> ws) {
    if (biasCol_ == -1) {
        // TODO
        throw std::runtime_error("provide bias col!");
    }

    auto root = std::make_shared<LinearObliviousTreeLeafV2>(bds, grid_, l2reg_, traceReg_, maxDepth_ + 1, 1, -1/*,
                                                            stats_[curLeavesCoord_].data()*/);

    usedFeatures_.insert(biasCol_);
    usedFeaturesOrdered_.push_back(biasCol_);
    root->usedFeatures_.insert(biasCol_);
    root->usedFeaturesInOrder_.push_back(biasCol_);

    parallelFor(0, nSamples_, [&](int blockId, int i) {
        auto& x = curX_[blockId];
        ds.fillSample(i, usedFeaturesOrdered_, x);

        auto bins = bds.sampleBins(i); // todo cache it somehow?
        float y = ys[i];
        float w = ws[i];

        for (int fId = 0; fId < fCount_; ++fId) {
            int offset = (int)binOffsets_[fId];
            int bin = offset + bins[fId];
            stats_[2 + blockId][0][bin].addFullCorrelation(x.data(), y, w);
        }
    });

    parallelFor(0, totalBins_, [&](int bin) {
        for (int blockId = 0; blockId < nThreads_; ++blockId) {
            root->hist_->addBinStat(bin, stats_[2 + blockId][0][bin]);
        }
    });

    root->hist_->prefixSumBins();

    return root;
}

void GreedyLinearObliviousTreeLearnerV2::computeNewCorrelations(const DataSet& ds, BinarizedDataSet& bds,
                                                                VecRef<float> ys, VecRef<float> ws, int nUsedFeatures) {
    memset(h_XTX_.data_ptr(), 0, h_XTX_.nbytes());
    memset(h_XTy_.data_ptr(), 0, h_XTy_.nbytes());

    parallelFor(0, nSamples_, [&](int blockId, int sampleId) {
        auto bins = bds.sampleBins(sampleId);
        auto& x = curX_[blockId];
        ds.fillSample(sampleId, usedFeaturesOrdered_, x);
        unsigned int lId = leafId_[sampleId];

        float y = ys[sampleId];
        float w = ws[sampleId];

        for (int fId = 0; fId < fCount_; ++fId) {
            auto origFId = grid_->origFeatureIndex(fId);
            if (usedFeatures_.count(origFId) != 0) continue;

            int bin = binOffsets_[fId] + bins[fId];

            float fVal = ds.fVal(sampleId, origFId);

            for (unsigned int i = 0; i < nUsedFeatures; ++i) {
                h_XTX_ref_[blockId][lId][bin][i] += x[i] * fVal * w;
            }
            h_XTX_ref_[blockId][lId][bin][nUsedFeatures] += fVal * fVal * w;
            h_XTy_ref_[blockId][lId][bin] += fVal * y * w;
        }
    });

    std::cout << 2.5 << std::endl;

    // todo change order?
    parallelFor(0, fCount_, [&](int fId) {
        auto origFId = grid_->origFeatureIndex(fId);
        if (usedFeatures_.count(origFId) != 0) return;

        for (int thId = 0; thId < nThreads_; ++thId) {
            for (int lId = 0; lId < (int) leaves_.size(); ++lId) {
                for (int localBinId = 0; localBinId <= (int) grid_->conditionsCount(fId); ++localBinId) {
                    int bin = binOffsets_[fId] + localBinId;
                    if (localBinId != 0) {
                        for (unsigned int i = 0; i <= nUsedFeatures; ++i) {
                            h_XTX_ref_[thId][lId][bin][i] += h_XTX_ref_[thId][lId][bin - 1][i];
                        }
                        h_XTy_ref_[thId][lId][bin] += h_XTy_ref_[thId][lId][bin - 1];
                    }
                    leaves_[lId]->hist_->addNewCorrelation(bin, h_XTX_ref_[thId][lId][bin].data(),
                                                          h_XTy_ref_[thId][lId][bin]);
                }
            }
        }
    }, true);
}


std::tuple<int, int, double> GreedyLinearObliviousTreeLearnerV2::findBestSplit() {
    double bestSplitScore = 1e9;
    int splitFId = -1;
    int splitCond = -1;

    std::vector<std::vector<double>> splitScores;
    for (int fId = 0; fId < fCount_; ++fId) {
        std::vector<double> fSplitScores;
        fSplitScores.resize(grid_->conditionsCount(fId), 0.0);
        splitScores.emplace_back(std::move(fSplitScores));
    }

    parallelFor(0, fCount_, [&](int fId) {
        parallelFor<1>(0, grid_->conditionsCount(fId), [&](int cond) {
            for (auto &l : leaves_) {
                splitScores[fId][cond] += l->splitScore(fId, cond);
            }
        });
    });

    for (int fId = 0; fId < fCount_; ++fId) {
        for (int64_t cond = 0; cond < grid_->conditionsCount(fId); ++cond) {
            double sScore = splitScores[fId][cond];
            if (sScore < bestSplitScore) {
                bestSplitScore = sScore;
                splitFId = fId;
                splitCond = cond;
            }
//                std::cout << "fId=" << fId << ", cond=" << cond << ", score=" << sScore << std::endl;
        }
    }

    return std::make_tuple(splitFId, splitCond, bestSplitScore);
}

void GreedyLinearObliviousTreeLearnerV2::cacheDs(const DataSet &ds) {

    // TODO this "caching" prevents from using bootstrap and rsm, but at least makes default boosting faster for now...

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

    nSamples_ = ds.samplesCount();

    fullUpdate_.resize(1 << maxDepth_, false);
    samplesLeavesCnt_.resize(1 << maxDepth_, 0);

    for (int i = 0; i < nThreads_; ++i) {
        curX_.emplace_back(maxDepth_ + 2, 0.0);
    }

    statsV_.clear();

    for (int thId = 0; thId < nThreads_ + 2; ++thId) {
        for (int lId = 0; lId < 1 << maxDepth_; ++lId) {
            for (int bin = 0; bin < totalBins_; ++bin) {
//                float* xtxPos = statsData_XTX_ref_[thId][lId][bin].data();
//                float* xtyPos = statsData_XTy_ref_[thId][lId][bin].data();
//                float* wPos = statsData_weight_ref_[thId][lId].data() + bin;
//                float* sumYPos = statsData_sumY_ref_[thId][lId].data() + bin;
//                float* sumY2Pos = statsData_sumY2_ref_[thId][lId].data() + bin;
//                float* sumXPos = statsData_sumX_ref_[thId][lId][bin].data();
//                BinStat::BinStatData data(xtxPos, xtyPos, wPos, sumYPos, sumY2Pos, sumXPos);
                statsV_.emplace_back(BinStat(maxDepth_ + 2, 1));
            }
        }
    }


    statsIdxs_ = multi_dim_array_idxs({nThreads_ + 2, 1 << maxDepth_, totalBins_});
    stats_ = MultiDimArray<3, BinStat>(statsV_.data(), &statsIdxs_, 0);

    isDsCached_ = true;
}

void GreedyLinearObliviousTreeLearnerV2::resetState() {
    std::cout << "reset 1" << std::endl;

    leaves_ = std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>>();

    usedFeatures_.clear();
    usedFeaturesOrdered_.clear();
    leafId_.clear();
    leafId_.resize(nSamples_, 0);

    std::cout << "reset 2" << std::endl;

//    memset(statsData_XTX_ref_.data(), 0, statsData_XTX_.nbytes());
//    memset(statsData_XTy_ref_.data(), 0, statsData_XTy_.nbytes());
//    memset(statsData_weight_ref_.data(), 0, statsData_weight_.nbytes());
//    memset(statsData_sumY_ref_.data(), 0, statsData_sumY_.nbytes());
//    memset(statsData_sumY2_ref_.data(), 0, statsData_sumY2_.nbytes());
//    memset(statsData_sumX_ref_.data(), 0, statsData_sumX_.nbytes());

    curLeavesCoord_ = 0;

    std::cout << "reset 3" << std::endl;

    // TODO only reset filled size?

    std::cout << "contig: " << h_XTX_.is_contiguous() << ", " << h_XTy_.is_contiguous() << std::endl;
    std::cout << nThreads_ << ", " << (1 << maxDepth_) << ", " << totalBins_ << std::endl;

//    std::cout << "statsData_XTX size = " << statsData_XTX_.nbytes() / sizeof(float) << std::endl;
//    std::cout << "statsData_XTy size = " << statsData_XTy_.nbytes() / sizeof(float) << std::endl;
//    std::cout << "statsData_weight size = " << statsData_weight_.nbytes() / sizeof(float) << std::endl;

    std::cout << "reset 5" << std::endl;
}

double LinearObliviousTreeV2::value(const ConstVecRef<float>& x) const {
    unsigned int lId = 0;

    for (int i = 0; i < splits_.size(); ++i) {
        const auto &s = splits_[i];
        auto fId = std::get<0>(s);
        auto condId = std::get<1>(s);

        const auto border = grid_->condition(fId, condId);
        const auto val = x[grid_->origFeatureIndex(fId)];
        if (val > border) {
            lId |= 1U << (splits_.size() - i - 1);
        }
    }

    return scale_ * leaves_[lId]->value(x);
}

void LinearObliviousTreeV2::applyToBds(const BinarizedDataSet& bds, Mx to, ApplyType type) const {
    const auto& ds = bds.owner();
    const uint64_t sampleDim = ds.featuresCount();
    const uint64_t targetDim = to.xdim();

    ConstVecRef<float> dsRef = ds.samplesMx().arrayRef();
    VecRef<float> toRef = to.arrayRef();

    uint64_t xSliceStart = 0;
    uint64_t toSliceStart = 0;

    for (uint64_t i = 0; i < ds.samplesCount(); ++i) {
        ConstVecRef<float> x = dsRef.slice(xSliceStart, sampleDim);
        VecRef<float> y = toRef.slice(toSliceStart, targetDim);

        switch (type) {
        case ApplyType::Append:
            y[0] += value(x);
            break;
        case ApplyType::Set:
        default:
            y[0] = value(x);
        }

        xSliceStart += sampleDim;
        toSliceStart += targetDim;
    }
}

void LinearObliviousTreeV2::appendTo(const Vec &x, Vec to) const {
    to += value(x.arrayRef());
}

double LinearObliviousTreeV2::value(const Vec &x) {
    return value(x.arrayRef());
}

void LinearObliviousTreeV2::grad(const Vec &x, Vec to) {
    throw std::runtime_error("Unimplemented");
}
