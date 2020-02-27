#pragma once

#include "grid.h"
#include "dataset.h"
#include <torch/torch.h>
#include <core/buffer.h>
#include <util/array_ref.h>
#include <util/parallel_executor.h>

struct FeaturesBundle {
    int32_t firstFeature_ = 0;
    int32_t lastFeature_ = 0;
    int32_t groupOffset_ = 0;

    int32_t groupSize() const {
        return lastFeature_ - firstFeature_;
    };
};

class BinarizedDataSet;
using BinarizedDataSetPtr = std::unique_ptr<BinarizedDataSet>;

class BinarizedDataSet : public Object {
public:

    GridPtr gridPtr() const {
        return grid_;
    }

    const Grid& grid() const {
        return *grid_;
    }

    int64_t groupCount() const {
        return groups_.size();
    }


    const FeaturesBundle& featuresBundle(int64_t groupIdx) const {
        return groups_[groupIdx];
    }

    ConstVecRef<uint8_t> group(int64_t groupIdx) const {
       return ConstVecRef<uint8_t>(data_.arrayRef().data() + groups_[groupIdx].groupOffset_ * samplesCount_,
                                     groups_[groupIdx].groupSize()  * samplesCount_);
    }


    int64_t samplesCount() const {
        return samplesCount_;
    }

    std::vector<uint8_t> sampleBins(int64_t sampleId) {
        std::vector<uint8_t> res;
        for (int fId = 0; fId < (int)grid_->nzFeaturesCount(); ++fId) {
            int64_t groupIdx = featureToGroup_.at(fId);
            const auto& groupInfo = groups_[groupIdx];
            auto groupBundle = group(groupIdx);
            const int64_t fIndexInGroup = fId - groupInfo.firstFeature_;

            res.push_back(groupBundle[sampleId * groupInfo.groupSize() + fIndexInGroup]);
        }
        return res;
    }

    template <class Visitor>
    void visitFeature(int64_t fIndex, Visitor&& visitor, bool parallel = false) const {
        int64_t groupIdx =  featureToGroup_.at(fIndex);
        const auto& groupInfo = groups_[groupIdx];
        auto groupBundle = group(groupIdx);
        const int64_t fIndexInGroup = fIndex - groupInfo.firstFeature_;
        if (parallel) {
            parallelFor(0, samplesCount_, [&](int blockId, int64_t i) {
                visitor(blockId, i, groupBundle[i * groupInfo.groupSize() + fIndexInGroup]);
            });
        } else {
            for (int64_t i = 0; i < samplesCount_; ++i) {
                visitor(0, i, groupBundle[i * groupInfo.groupSize() + fIndexInGroup]);
            }
        }
    }


    template <class Visitor>
    void visitGroups(Visitor&& visitor) const {
        for (int32_t groupId = 0; groupId < groups_.size(); ++groupId) {
            const auto& groupInfo = groups_[groupId];
            auto groupBundle = group(groupId);
            visitor(groupInfo, groupBundle);
        }
    }


    template <class Visitor>
    void visitFeature(int64_t fIndex, ConstVecRef<int32_t> indices, Visitor&& visitor, bool parallel = false) const {
        int64_t groupIdx =  featureToGroup_.at(fIndex);
        const auto& groupInfo = groups_[groupIdx];
        auto groupBundle = group(groupIdx);
        const int64_t fIndexInGroup = fIndex - groupInfo.firstFeature_;
        if (parallel) {
            parallelFor(0, indices.size(), [&](int blockId, int64_t i) {
                visitor(blockId, i, groupBundle[indices[i] * groupInfo.groupSize() + fIndexInGroup]);
            });
        } else {
            for (int64_t i = 0; i < indices.size(); ++i) {
                visitor(0, i, groupBundle[indices[i] * groupInfo.groupSize() + fIndexInGroup]);
            }
        }
    }


    ConstVecRef<int32_t> binOffsets() const {
        return grid_->binOffsets();
    }

    int32_t totalBins() const {
        return grid_->totalBins();
    }

    const DataSet& owner() const {
        return owner_;
    }

private:
    VecRef<uint8_t> group(int64_t groupIdx) {
        return VecRef<uint8_t>(data_.arrayRef().data() + groups_[groupIdx].groupOffset_ * samplesCount_,
                                 groups_[groupIdx].groupSize() * samplesCount_);
    }


    template <class Visitor>
    void updateLineForGroup(int64_t groupIdx, int64_t line, Visitor&& updater) {
        ConstVecRef<int32_t> featureIds = groupToFeatures[groupIdx];
        VecRef<uint8_t> groupRef = group(groupIdx);
        const auto groupSize = groups_[groupIdx].groupSize();
        auto lineBins = VecRef<uint8_t>(groupRef.data() + line * groupSize, groupSize);
        updater(featureIds, lineBins);
    }


    BinarizedDataSet(const DataSet& owner,
        GridPtr grid,
        int64_t samplesCount,
        std::vector<FeaturesBundle>&& groups)
        : owner_(owner)
        , grid_(std::move(grid))
        , samplesCount_(samplesCount)
        , groups_(std::move(groups))
        , data_(Buffer<uint8_t>::create(samplesCount * (groups_.back().groupOffset_ + groups_.back().groupSize()))) {
        data_.fill(0);
        featureToGroup_.resize(grid_->nzFeaturesCount());
        groupToFeatures.resize(groups_.size());

        for (int32_t groupIdx = 0; groupIdx < groups_.size(); ++groupIdx) {
            const auto& group  = groups_[groupIdx];
            for (int32_t f=  group.firstFeature_; f < group.lastFeature_; ++f) {
                featureToGroup_[f] = groupIdx;
                groupToFeatures[groupIdx].push_back(f);
            }
        }
    }

    friend BinarizedDataSetPtr binarize(const DataSet& ds, GridPtr& grid, int32_t maxGroupSize);

private:
    const DataSet& owner_;
    GridPtr grid_;
    int64_t samplesCount_;
    std::vector<FeaturesBundle> groups_;
    std::vector<int32_t> featureToGroup_;
    std::vector<std::vector<int32_t>> groupToFeatures;
    Buffer<uint8_t> data_;
};


void createGroups(const Grid& grid, int32_t maxGroupSize, std::vector<FeaturesBundle>* bundles);

inline std::vector<FeaturesBundle> createGroups(const Grid& grid, int32_t maxGroupSize) {
    std::vector<FeaturesBundle> groups;
    createGroups(grid, maxGroupSize, &groups);
    return groups;
}

BinarizedDataSetPtr binarize(const DataSet& ds, GridPtr& grid, int32_t maxGroupSize = 16);



inline const BinarizedDataSet& cachedBinarize(const DataSet& ds, GridPtr grid, int32_t maxGroupSize = 16) {
    return ds.computeOrGet<Grid, BinarizedDataSet>(std::move(grid), [&](const DataSet& ds, GridPtr ptr) -> std::unique_ptr<BinarizedDataSet> {
        auto start = std::chrono::system_clock::now();
        auto binarized = binarize(ds, ptr, maxGroupSize);
        std::cout << "binarization time " << std::chrono::duration<double>(std::chrono::system_clock::now() - start).count()
                  << std::endl;
        return binarized;
    });
}


