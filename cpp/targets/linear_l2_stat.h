#pragma once

#include <vector>
#include <stdexcept>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include "additive_statistics.h"
#include <util/array_ref.h>

#include <iostream>

struct LinearL2CorStatOpParams {
    float fVal = 0.;
};

struct LinearL2CorStatTypeTraits {
    using ImplSampleType = const float*;
    using ImplTargetType = float;
    using ImplWeightType = float;
};

struct LinearL2CorStat : public AdditiveStatistics<LinearL2CorStat,
        LinearL2CorStatTypeTraits, LinearL2CorStatOpParams> {
    explicit LinearL2CorStat(int size);

    LinearL2CorStat& appendImpl(const LinearL2CorStat& other, const LinearL2CorStatOpParams& opParams);
    LinearL2CorStat& removeImpl(const LinearL2CorStat& other, const LinearL2CorStatOpParams& opParams);

    LinearL2CorStat& appendImpl(SampleType x, TargetType y, WeightType weight, const LinearL2CorStatOpParams& opParams);
    LinearL2CorStat& removeImpl(SampleType x, TargetType y, WeightType weight, const LinearL2CorStatOpParams& opParams);

    int size_;
    std::vector<float> xxt;
    float xy;
};



struct LinearL2StatOpParams {
    int opSize = -1; // will be treated as filledSize
    int shift = 0;

    enum VecAddMode {
        NewCorrelation,
        FullCorrelation
    } vecAddMode = NewCorrelation;
};

struct LinearL2StatTypeTraits {
    using ImplSampleType = const float*; // TODO CorStat
    using ImplTargetType = float;
    using ImplWeightType = float;
};

struct LinearL2Stat : public AdditiveStatistics<LinearL2Stat, LinearL2StatTypeTraits, LinearL2StatOpParams> {
    using EMx = Eigen::MatrixXd;

    LinearL2Stat(int size, int filledSize);

    void reset();

    void setFilledSize(int filledSize) {
        filledSize_ = filledSize;
        maxUpdatedPos_ = filledSize_;
    }
//
//    [[nodiscard]] int filledSize() const {
//        return filledSize_;
//    }
//
//    [[nodiscard]] int mxSize() const {
//        return maxUpdatedPos_;
//    }
//
    void addNewCorrelation(SampleType xtx, TargetType xty, WeightType w, int shift = 0);
    void addFullCorrelation(SampleType x, TargetType y, WeightType w);

    LinearL2Stat& appendImpl(const LinearL2Stat& other, const LinearL2StatOpParams& opParams);
    LinearL2Stat& removeImpl(const LinearL2Stat& other, const LinearL2StatOpParams& opParams);

    LinearL2Stat& appendImpl(SampleType x, TargetType y, WeightType weight, const LinearL2StatOpParams& opParams);
    LinearL2Stat& removeImpl(SampleType x, TargetType y, WeightType weight, const LinearL2StatOpParams& opParams);

    void fillXTX(EMx& XTX) const;
    [[nodiscard]] EMx getXTX() const;

    void fillXTy(EMx& XTy) const;
    [[nodiscard]] EMx getXTy() const;

    void fillSumX(EMx& sumX) const;
    [[nodiscard]] EMx getSumX() const;

    [[nodiscard]] EMx getWHat(double l2reg) const;

    int size_;
    int filledSize_;
    int maxUpdatedPos_;

    float w_;
    float trace_;
    float sumY_;
    float sumY2_;
    std::vector<float> xtx_;
    std::vector<float> xty_;
    std::vector<float> sumX_;
};

struct LinearL2GridStatOpParams : public LinearL2StatOpParams {
    int bin = -1;
};

class LinearL2GridStat : public AdditiveStatistics<LinearL2GridStat, LinearL2StatTypeTraits, LinearL2GridStatOpParams> {
public:
    LinearL2GridStat(int nBins, int size, int filledSize);

    void reset();

    void setFilledSize(int filledSize);

    LinearL2GridStat& appendImpl(const LinearL2GridStat& other, const LinearL2GridStatOpParams& opParams);
    LinearL2GridStat& removeImpl(const LinearL2GridStat& other, const LinearL2GridStatOpParams& opParams);

    LinearL2GridStat& appendImpl(SampleType x, TargetType y, WeightType weight, const LinearL2GridStatOpParams& opParams);
    LinearL2GridStat& removeImpl(SampleType x, TargetType y, WeightType weight, const LinearL2GridStatOpParams& opParams);

    LinearL2Stat& getBinStat(int bin) {
        return stats_[bin];
    }

private:
    int nBins_;
    int size_;
    int filledSize_;

    std::vector<LinearL2Stat> stats_;
};
