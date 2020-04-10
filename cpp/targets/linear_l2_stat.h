#pragma once

#include <vector>
#include <stdexcept>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include "additive_statistics.h"
#include <util/array_ref.h>

#include <iostream>

struct LinearL2CorStatTypeTraits {
    using ImplFeatureType = float;
    using ImplSampleType = const ImplFeatureType*;
    using ImplTargetType = float;
    using ImplWeightType = float;
};

struct LinearL2CorStatOpParams {
    LinearL2CorStatTypeTraits::ImplFeatureType fVal = 0.;
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
    float sumX;

    friend std::ostream& operator<<(std::ostream& os, const LinearL2CorStat& s);
};

inline std::ostream& operator<<(std::ostream& os, const LinearL2CorStat& s) {
    os << "xy=" << s.xy << ", sumX=" << s.sumX << ", xxt=[";
    for (int i = 0; i < s.size_; ++i) {
        os << s.xxt[i] << " ";
    }
    os << "]";
    return os;
}


struct LinearL2StatOpParams {
    int opSize = -1;
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
    // TODO: using y as xty and w as sumX. Bad interface :/
    void addNewCorrelation(SampleType xtx, TargetType xty, WeightType sumX, int shift = 0);
    void addFullCorrelation(SampleType x, TargetType y, WeightType w);

    LinearL2Stat& appendImpl(const LinearL2Stat& other, const LinearL2StatOpParams& opParams);
    LinearL2Stat& removeImpl(const LinearL2Stat& other, const LinearL2StatOpParams& opParams);

    LinearL2Stat& appendImpl(SampleType x, TargetType y, WeightType weight, const LinearL2StatOpParams& opParams);
    LinearL2Stat& removeImpl(SampleType x, TargetType y, WeightType weight, const LinearL2StatOpParams& opParams);

    void fillXTX(EMx& XTX, double l2reg = 0.) const;
    [[nodiscard]] EMx getXTX(double l2reg = 0., int size = 0) const;

    void fillXTy(EMx& XTy) const;
    [[nodiscard]] EMx getXTy(int size = 0) const;

    void fillSumX(EMx& sumX) const;
    [[nodiscard]] EMx getSumX(int size = 0) const;

    [[nodiscard]] EMx getWHat(double l2reg, int size = 0) const;

    int size_;
    int filledSize_;
    int maxUpdatedPos_;

    float w_;
    float sumY_;
    float sumY2_;
    std::vector<float> xtx_;
    std::vector<float> xty_;
    std::vector<float> sumX_;

    friend std::ostream& operator<<(std::ostream& os, const LinearL2Stat& s);
};

inline std::ostream& operator<<(std::ostream& os, const LinearL2Stat& s) {
    os << "w=" << s.w_ << ", XTX=" << s.getXTX() << ", XTy=" << s.getXTy() << ", sumY=" << s.sumY_;
    return os;
}

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
