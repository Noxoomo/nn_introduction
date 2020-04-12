#pragma once

#include <cstdint>
#include <vector>
#include <core/vec.h>
#include <core/object.h>
#include <core/buffer.h>

struct BinaryFeature {
    int32_t featureId_ = 0;
    int32_t conditionId_ = 0;

    BinaryFeature(int32_t featureId_, int32_t binaryFeatureId_)
        : featureId_(featureId_)
        , conditionId_(binaryFeatureId_) {

    }

    BinaryFeature(const BinaryFeature& other) = default;
};

struct Feature {
    int32_t featureId_ = 0;
    int32_t conditionsCount_ = 0;
    int32_t origFeatureId_  = 0;

    Feature(int32_t featureId_, int32_t binFeatures, int32_t origFeatureId)
        : featureId_(featureId_)
        , conditionsCount_(binFeatures)
        , origFeatureId_(origFeatureId){

    }

    Feature(const Feature& other) = default;

};

class Grid : public UuidHolder {
public:

    int64_t origFeaturesCount() const {
        return fCount_;
    }

    BinaryFeature binFeature(int32_t seqIdx) const {
        return binFeatures_[seqIdx];
    }


    int64_t nzFeaturesCount() const {
        return features_.size();
    }

    int64_t origFeatureIndex(int64_t localIdx) const {
        return features_[localIdx].origFeatureId_;
    }

    double condition(int64_t fIndex, int64_t binIndex) const {
        return borders_[fIndex][binIndex];
    }

    double condition(int64_t binIdx) const {
        auto binFeature = this->binFeature(binIdx);
        return condition(binFeature.featureId_, binFeature.conditionId_);
    }

    ConstVecRef<Feature> nzFeatures() const {
        return ConstVecRef<Feature>(features_);
    }

    ConstVecRef<float> borders(int64_t fIndex) const {
        return borders_[fIndex];
    }

    void binarize(const Vec& x, Buffer<uint8_t>& to) const;


    int64_t binFeaturesCount() const {
        return binFeatures_.size();
    }

    ConstVecRef<int64_t> binFeatureOffsets() const {
        return binFeatureOffsets_;
    }

    int64_t conditionsCount(int32_t fIndex) const {
        return features_[fIndex].conditionsCount_;
    }

    int32_t totalBins() const {
        return binOffsets_.back();
    }

    ConstVecRef<int32_t> binOffsets() const {
        return binOffsets_;
    }

    void binarize(ConstVecRef<float> row, VecRef<uint8_t> dst) const;
//
//    void binarizeColumn(int32_t fIndex, const Vec& column, torch::Tensor* dst) const;


    Grid(
        int64_t fCount,
        std::vector<BinaryFeature>&& binFeatures,
        std::vector<Feature>&& features,
        std::vector<std::vector<float>>&& borders)
        : fCount_(fCount)
        , binFeatures_(std::move(binFeatures))
        , features_(std::move(features))
        , borders_(std::move(borders)) {
        binFeatureOffsets_.resize(features_.size());
        for (int64_t i = 1; i < features_.size(); ++i) {
            binFeatureOffsets_[i] = binFeatureOffsets_[i - 1] + features_[i - 1].conditionsCount_;
        }

        int32_t cursor = 0;
        for (int32_t fIndex = 0; fIndex < nzFeaturesCount(); ++fIndex) {
            binOffsets_.push_back(cursor);
            cursor += conditionsCount(fIndex) + 1;
        }
        binOffsets_.push_back(cursor);
    }

private:
    int64_t fCount_;
    std::vector<int32_t> binOffsets_;
    std::vector<BinaryFeature> binFeatures_;
    std::vector<Feature> features_;
    std::vector<std::vector<float>> borders_;
    std::vector<int64_t> binFeatureOffsets_;
};


inline int32_t computeBin(float val, ConstVecRef<float> borders) {
    int32_t bin = 0;
    while (bin < borders.size() && borders[bin] < val) {
        ++bin;
    }
    return bin;
}


using GridPtr = std::shared_ptr<Grid>;


