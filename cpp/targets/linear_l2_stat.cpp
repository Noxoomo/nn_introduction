#include "linear_l2_stat.h"


namespace {
    void vectorizedAdd(float* dst, const float* src, int size) {
        int i;
        for (i = 0; i <= size - 4; i += 4) {
            for (int j = i; j < i + 4; ++j) {
                dst[j] += src[j];
            }
        }
        for (; i < size; ++i) {
            dst[i] += src[i];
        }
    }

    void vectorizedScaleAdd(float* dst, const float* src, float m, int size) {
        int i;
        for (i = 0; i <= size - 4; i += 4) {
            for (int j = i; j < i + 4; ++j) {
                dst[j] += m * src[j];
            }
        }
        for (; i < size; ++i) {
            dst[i] += m * src[i];
        }
    }

    void vectorizedRm(float* dst, const float* src, int size) {
        int i;
        for (i = 0; i <= size - 4; i += 4) {
            for (int j = i; j < i + 4; ++j) {
                dst[j] -= src[j];
            }
        }
        for (; i < size; ++i) {
            dst[i] -= src[i];
        }
    }

    void vectorizedScaleRm(float* dst, const float* src, float m, int size) {
        int i;
        for (i = 0; i <= size - 4; i += 4) {
            for (int j = i; j < i + 4; ++j) {
                dst[j] -= m * src[j];
            }
        }
        for (; i < size; ++i) {
            dst[i] -= m * src[i];
        }
    }
}

// LinearL2CorStat


LinearL2CorStat::LinearL2CorStat(int size, int filledSize)
        : size_(size)
        , filledSize_(filledSize) {
    xxt.resize(size_, 0.);
    xy = 0.;
    sumX = 0.;
}

void LinearL2CorStat::reset() {
    std::fill_n(xxt.begin(), filledSize_, 0);
    xy = 0;
    sumX = 0;
}

void LinearL2CorStat::setFilledSize(int filledSize) {
    filledSize_ = filledSize;
}

LinearL2CorStat& LinearL2CorStat::appendImpl(const LinearL2CorStat &other,
                                             const LinearL2CorStatOpParams &opParams) {
//    for (int i = 0; i < filledSize_; ++i) {
//        xxt[i] += other.xxt[i];
//    }
    vectorizedAdd(xxt.data(), other.xxt.data(), filledSize_);
    xy += other.xy;
    sumX += other.sumX;
}

LinearL2CorStat& LinearL2CorStat::removeImpl(const LinearL2CorStat &other,
                                             const LinearL2CorStatOpParams &opParams) {
//    for (int i = 0; i < filledSize_; ++i) {
//        xxt[i] -= other.xxt[i];
//    }
    vectorizedRm(xxt.data(), other.xxt.data(), filledSize_);
    xy -= other.xy;
    sumX -= other.sumX;
}

LinearL2CorStat& LinearL2CorStat::appendImpl(const float* x, float y, float weight,
                                             const LinearL2CorStatOpParams &opParams) {
    const float wf = weight * opParams.fVal;
//    for (int i = 0; i < filledSize_ - 1; ++i) {
//        xxt[i] += x[i] * wf;
//    }
    vectorizedScaleAdd(xxt.data(), x, wf, filledSize_ - 1);
    xxt[filledSize_ - 1] += opParams.fVal * wf;
    xy += y * wf;
    sumX += wf;
}

LinearL2CorStat& LinearL2CorStat::removeImpl(const float* x, float y, float weight,
                                             const LinearL2CorStatOpParams &opParams) {
    const float wf = weight * opParams.fVal;
//    for (int i = 0; i < filledSize_ - 1; ++i) {
//        xxt[i] -= x[i] * wf;
//    }
    vectorizedScaleRm(xxt.data(), x, wf, filledSize_ - 1);
    xxt[filledSize_ - 1] -= opParams.fVal * wf;
    xy -= y * wf;
    sumX -= wf;
}


// LinearL2Stat


LinearL2Stat::LinearL2Stat(int size, int filledSize)
        : size_(size)
        , filledSize_(filledSize) {
    maxUpdatedPos_ = filledSize_;

    w_ = 0;
    sumY_ = 0;
    sumY2_ = 0;
    xtx_.resize(size * (size + 1) / 2, 0.);
    xty_.resize(size, 0.);
    sumX_.resize(size, 0.);
}

void LinearL2Stat::reset() {
    w_ = 0;
    sumY_ = 0;
    sumY2_ = 0;

    std::fill_n(xtx_.begin(), (maxUpdatedPos_ * (maxUpdatedPos_ + 1) / 2), 0);
    std::fill_n(xty_.begin(), maxUpdatedPos_, 0);
    std::fill_n(sumX_.begin(), maxUpdatedPos_, 0);

    filledSize_ = 0;
    maxUpdatedPos_ = 0;
}

void LinearL2Stat::addNewCorrelation(const float* xtx, float xty, float sumX, int shift)  {
    // TODO: using y as xty and w as sumX. Bad interface :/

    const int corPos = filledSize_ + shift;

    int pos = corPos * (corPos + 1) / 2;
//    for (int i = 0; i <= corPos; ++i) {
//        xtx_[pos + i] += xtx[i];
//    }
    vectorizedAdd(xtx_.data() + pos, xtx, corPos + 1);
    xty_[corPos] += xty;
    sumX_[corPos] += sumX;
    maxUpdatedPos_ = std::max(maxUpdatedPos_, corPos + 1);
}

void LinearL2Stat::addFullCorrelation(const float* x, float y, float w) {
    w_ += w;
    float yw = y * w;
    sumY_ += yw;
    sumY2_ += yw * y;

    int pos = 0;
    for (int i = 0; i < filledSize_; ++i) {
        float xiw = x[i] * w;
        for (int j = 0; j < i + 1; ++j) {
            xtx_[pos + j] += xiw * x[j];
        }
        pos += i + 1;
    }

    vectorizedScaleAdd(sumX_.data(), x, w, filledSize_);
    vectorizedScaleAdd(xty_.data(), x, yw, filledSize_);
}

LinearL2Stat& LinearL2Stat::appendImpl(const LinearL2Stat &other, const LinearL2StatOpParams &opParams) {
    w_ += other.w_;
    sumY_ += other.sumY_;
    sumY2_ += other.sumY2_;

    int size = opParams.opSize;
    if (size < 0) {
        size = std::min(maxUpdatedPos_, other.maxUpdatedPos_); // TODO filled or updated?
    }

//    int pos = 0;
//    for (int i = 0; i < size; ++i) {
//        for (int j = 0; j < i + 1; ++j) {
//            xtx_[pos + j] += other.xtx_[pos + j];
//        }
//        pos += i + 1;
//        xty_[i] += other.xty_[i];
//        sumX_[i] += other.sumX_[i];
//    }

    int pos = size * (size + 1) / 2;
    vectorizedAdd(xtx_.data(), other.xtx_.data(), pos);

    vectorizedAdd(xty_.data(), other.xty_.data(), size);
    vectorizedAdd(sumX_.data(), other.sumX_.data(), size);


    return *this;
}

LinearL2Stat& LinearL2Stat::removeImpl(const LinearL2Stat &other, const LinearL2StatOpParams &opParams) {
    w_ -= other.w_;
    sumY_ -= other.sumY_;
    sumY2_ -= other.sumY2_;

    int size = opParams.opSize;
    if (size < 0) {
        size = std::min(maxUpdatedPos_, other.maxUpdatedPos_); // TODO filled or updated?
    }

//    int pos = 0;
//    for (int i = 0; i < size; ++i) {
//        for (int j = 0; j < i + 1; ++j) {
//            xtx_[pos + j] -= other.xtx_[pos + j];
//        }
//        pos += i + 1;
//        xty_[i] -= other.xty_[i];
//        sumX_[i] -= other.sumX_[i];
//    }

    int pos = size * (size + 1) / 2;
    vectorizedRm(xtx_.data(), other.xtx_.data(), pos);

    vectorizedRm(xty_.data(), other.xty_.data(), size);
    vectorizedRm(sumX_.data(), other.sumX_.data(), size);

    return *this;
}

LinearL2Stat& LinearL2Stat::appendImpl(const float* x, float y, float weight,
                                       const LinearL2StatOpParams &opParams) {
    switch (opParams.vecAddMode) {
        case LinearL2StatOpParams::FullCorrelation:
            addFullCorrelation(x, y, weight);
            break;

        case LinearL2StatOpParams::NewCorrelation:
        default:
            addNewCorrelation(x, y, weight, opParams.shift);
            break;
    }

    return *this;
}

LinearL2Stat& LinearL2Stat::removeImpl(const float* x, float y, float weight,
                                       const LinearL2StatOpParams &opParams) {
    throw std::runtime_error("Unimplemented");
//    appendImpl(x, y, -1 * weight, opParams);
}

void LinearL2Stat::fillXTX(LinearL2Stat::EMx& XTX, double l2reg) const {
    int basePos = 0;
    for (int i = 0; i < XTX.rows(); ++i) {
        for (int j = 0; j < i + 1; ++j) {
            XTX(i, j) = xtx_[basePos + j];
            XTX(j, i) = xtx_[basePos + j];
        }
        XTX(i, i) += l2reg;
        basePos += i + 1;
    }
}

LinearL2Stat::EMx LinearL2Stat::getXTX(double l2reg, int size) const {
    if (size == 0) {
        size = maxUpdatedPos_;
    }
    EMx res(size, size);
    fillXTX(res, l2reg);
    return res;
}

void LinearL2Stat::fillXTy(EMx &XTy) const {
    for (int i = 0; i < XTy.rows(); ++i) {
        XTy(i, 0) = xty_[i];
    }
}

LinearL2Stat::EMx LinearL2Stat::getXTy(int size) const {
    if (size == 0) {
        size = maxUpdatedPos_;
    }
    EMx res(size, 1);
    fillXTy(res);
    return res;
}

void LinearL2Stat::fillSumX(EMx& sumX) const {
    for (int i = 0; i < sumX.rows(); ++i) {
        sumX(i, 0) = sumX_[i];
    }
}

LinearL2Stat::EMx LinearL2Stat::getSumX(int size) const {
    if (size == 0) {
        size = maxUpdatedPos_;
    }
    EMx res(size, 1);
    fillSumX(res);
    return res;
}

LinearL2Stat::EMx LinearL2Stat::getWHat(double l2reg, int size) const {
    if (size == 0) {
        size = maxUpdatedPos_;
    }
    if (w_ < 1e-6) {
        auto w = EMx(size, 1);
        for (int i = 0; i < w.rows(); ++i) {
            w(i, 0) = 0;
        }
        return w;
    }

    EMx XTX = getXTX(l2reg, size);
    return XTX.inverse() * getXTy(size);
}

LinearL2GridStat::LinearL2GridStat(int nBins, int size, int filledSize)
        : nBins_(nBins)
        , size_(size)
        , filledSize_(filledSize) {
    stats_.resize(nBins, LinearL2Stat(size, filledSize));
}

void LinearL2GridStat::reset() {
    for (auto& stat : stats_) {
        stat.reset();
    }
}

void LinearL2GridStat::setFilledSize(int filledSize) {
    filledSize_ = filledSize;
    for (auto& stat : stats_) {
        stat.setFilledSize(filledSize_);
    }
}

LinearL2GridStat& LinearL2GridStat::appendImpl(const LinearL2GridStat& other, const LinearL2GridStatOpParams& opParams) {
    assert(nBins_ == other.nBins_);
    for (int i = 0; i < nBins_; ++i) {
        stats_[i].append(other.stats_[i], opParams);
    }
}

LinearL2GridStat& LinearL2GridStat::removeImpl(const LinearL2GridStat& other, const LinearL2GridStatOpParams& opParams) {
    assert(nBins_ == other.nBins_);
    for (int i = 0; i < nBins_; ++i) {
        stats_[i].remove(other.stats_[i], opParams);
    }
}

LinearL2GridStat& LinearL2GridStat::appendImpl(SampleType x, TargetType y, WeightType weight, const LinearL2GridStatOpParams& opParams) {
    int bin = opParams.bin;
    stats_[bin].append(x, y, weight, opParams);
}

LinearL2GridStat& LinearL2GridStat::removeImpl(SampleType x, TargetType y, WeightType weight, const LinearL2GridStatOpParams& opParams) {
    int bin = opParams.bin;
    stats_[bin].remove(x, y, weight, opParams);
}
