#pragma once

#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>


class BinStat {
public:

//    struct BinStatData {
//        float* xtx;
//        float* xty;
//        float* w;
//        float* sumY;
//        float* sumY2;
//        float* sumX;
//
//        BinStatData() = default;
//
//        BinStatData(float* xtx, float* xty, float* w, float* sumY, float* sumY2, float* sumX)
//                : xtx(xtx)
//                , xty(xty)
//                , w(w)
//                , sumY(sumY)
//                , sumY2(sumY2)
//                , sumX(sumX) {
//
//        }
//    };

    using EMx = Eigen::MatrixXd;
    using fType = double;

    explicit BinStat(int size, int filledSize)
            : size_(size)
            , filledSize_(filledSize) {
        maxUpdatedPos_ = filledSize_;

        w_ = 0;
        sumY_ = 0;
        sumY2_ = 0;
        trace_ = 0;
        xtx_.resize(size * (size + 1) / 2);
        xty_.resize(size);
        sumX_.resize(size);
    }

    void reset() {
        filledSize_ = 0;
        maxUpdatedPos_ = 0;

        w_ = 0;
        sumY_ = 0;
        sumY2_ = 0;
        trace_ = 0;
        memset(xtx_.data(), 0, xtx_.size() * sizeof(float));
        memset(xty_.data(), 0, xty_.size() * sizeof(float));
        memset(sumX_.data(), 0, sumX_.size() * sizeof(float));
    }

    void setFilledSize(int filledSize) {
        filledSize_ = filledSize;
        maxUpdatedPos_ = filledSize_;
    }

    [[nodiscard]] int filledSize() const {
        return filledSize_;
    }

    [[nodiscard]] int mxSize() const {
        return maxUpdatedPos_;
    }

    void addNewCorrelation(const float* xtx, float xty, int shift = 0) {
        const int corPos = filledSize_ + shift;

        // TODO Weights? (only relevant for the new objective)
        int pos = corPos * (corPos + 1) / 2;
        for (int i = 0; i <= corPos; ++i) {
            xtx_[pos + i] += xtx[i];
        }
        xty_[corPos] += xty;
        maxUpdatedPos_ = std::max(maxUpdatedPos_, corPos + 1);
    }

    void addFullCorrelation(const float* x, float y, float w) {
        int pos = 0;
        for (int i = 0; i < filledSize_; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                xtx_[pos + j] += x[i] * x[j] * w;
            }
            sumX_[i] += x[i] * w;
            xty_[i] += x[i] * y * w;
            pos += i + 1;
        }

        w_ += w;
        sumY_ += y * w;
        sumY2_ += y * y * w;
    }

    void fillXTX(EMx& XTX) const {
        int basePos = 0;
        for (int i = 0; i < maxUpdatedPos_; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                XTX(i, j) = xtx_[basePos + j];
                XTX(j, i) = xtx_[basePos + j];
            }
            basePos += i + 1;
        }
    }

    [[nodiscard]] EMx getXTX() const {
        EMx res(maxUpdatedPos_, maxUpdatedPos_);
        fillXTX(res);
        return res;
    }

    void fillXTy(EMx& XTy) const {
        for (int i = 0; i < maxUpdatedPos_; ++i) {
            XTy(i, 0) = xty_[i];
        }
    }

    [[nodiscard]] EMx getXTy() const {
        EMx res(maxUpdatedPos_, 1);
        fillXTy(res);
        return res;
    }

    void fillSumX(EMx& sumX) const {
        for (int i = 0; i < maxUpdatedPos_; ++i) {
            sumX(i, 0) = sumX_[i];
        }
    }

    [[nodiscard]] EMx getSumX() const {
        EMx res(maxUpdatedPos_, 1);
        fillSumX(res);
        return res;
    }

    [[nodiscard]] EMx getW(double l2reg) const {
        auto XTX = getXTX();

        if (w_ < 1e-6) {
            auto w = EMx(XTX.rows(), 1);
            for (int i = 0; i < w.rows(); ++i) {
                w(i, 0) = 0;
            }
            return w;
        }

        EMx XTX_r = XTX + DiagMx(XTX.rows(), l2reg);
        return XTX_r.inverse() * getXTy();
    }

    [[nodiscard]] float getWeight() const {
        return w_;
    }

    // This one DOES NOT add up new correlations
    BinStat& operator+=(const BinStat& s) {
        w_ += s.w_;
        sumY_ += s.sumY_;
        sumY2_ += s.sumY2_;
        trace_ += s.trace_;

        int size = std::min(filledSize_, s.filledSize_);

        int pos = 0;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                xtx_[pos + j] += s.xtx_[pos + j];
            }
            pos += i + 1;
            xty_[i] += s.xty_[i];
            sumX_[i] += s.sumX_[i];
        }
    }

    // This one DOES NOT subtract new correlations
    BinStat& operator-=(const BinStat& s) {
        w_ -= s.w_;
        sumY_ -= s.sumY_;
        sumY2_ -= s.sumY2_;
        trace_ -= s.trace_;

        int size = std::min(filledSize_, s.filledSize_);

        int pos = 0;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                xtx_[pos + j] -= s.xtx_[pos + j];
            }
            pos += i + 1;
            xty_[i] -= s.xty_[i];
            sumX_[i] -= s.sumX_[i];
        }
    }

    static float fitScore(const BinStat& bs, double l2Reg) {
//        float w = bs.w_;
//        float sumY = bs.sumY_;
//
//        if (w < 2) {
//            return 0;
//        }
//
//        EMx wHat = bs.getW(l2Reg);
//
//        EMx xty = bs.getXTy();
//        xty -= (sumY / w) * bs.getSumX();
////        xty *= 1.0 / w;
//
//        float reg = 1 + 0.005f * std::log(w + 1);
//
//        float scoreFromLinear = (xty.transpose() * wHat)(0, 0);
//        float scoreFromConst = (sumY * sumY) / w;
//        float targetValue = scoreFromConst + scoreFromLinear - l2Reg * (wHat.transpose() * wHat)(0, 0);
//
//        return -targetValue * reg;

        if (bs.getWeight() < 1e-5) {
            return 0;
        }

        EMx w = bs.getW(l2Reg);

        EMx c1(bs.getXTy().transpose() * w);
        c1 *= -2;
        assert(c1.rows() == 1 && c1.cols() == 1);

        EMx c2(w.transpose() * bs.getXTX() * w);
        assert(c2.rows() == 1 && c2.cols() == 1);

        EMx reg = w.transpose() * w * l2Reg;
        assert(reg.rows() == 1 && reg.cols() == 1);

        EMx res = c1 + c2 + reg;

        return res(0, 0);
    }

    [[nodiscard]] static inline EMx DiagMx(int dim, fType v) {
        EMx mx(dim, dim);
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                if (j == i) {
                    mx(i, j) = v;
                } else {
                    mx(i, j) = 0;
                }
            }
        }
        return mx;
    }

private:
    friend BinStat operator+(const BinStat& lhs, const BinStat& rhs);
    friend BinStat operator-(const BinStat& lhs, const BinStat& rhs);

private:
    int size_;
    int filledSize_;
    int maxUpdatedPos_;

    fType w_;
    fType sumY_;
    fType sumY2_;
    fType trace_;

    std::vector<fType> xtx_;
    std::vector<fType> xty_;
    std::vector<fType> sumX_;
};

inline BinStat operator+(const BinStat& lhs, const BinStat& rhs) {
    BinStat res(lhs);
    res += rhs;
    return res;
}

inline BinStat operator-(const BinStat& lhs, const BinStat& rhs) {
    BinStat res(lhs);
    res -= rhs;
    return res;
}
