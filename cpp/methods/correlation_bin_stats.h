#pragma once

#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>


class BinStat {
public:
    BinStat() = default;

    explicit BinStat(float* XTX_data, float* XTy_data, float* weight_data, int size, int filledSize)
            : XTX_(XTX_data)
            , XTy_(XTy_data)
            , weight_(weight_data)
            , size_(size)
            , filledSize_(filledSize) {
//        XTX_.resize(size * (size + 1) / 2, 0.0);
//        XTy_ = std::vector<float>(size, 0.0);
//        cnt_ = 0;
        trace_ = 0.0;
        maxUpdatedPos_ = filledSize_;
    }

    void reset() {
        *weight_ = 0;
        trace_ = 0;
        filledSize_ = 0;
        maxUpdatedPos_ = 0;
        memset(XTX_, 0, size_ * (size_ + 1) / 2 * sizeof(float));
        memset(XTy_, 0, size_ * sizeof(float));
    }

    void setFilledSize(int filledSize) {
        filledSize_ = filledSize;
        maxUpdatedPos_ = filledSize_;
    }

    int filledSize() const {
        return filledSize_;
    }

    int mxSize() const {
        return maxUpdatedPos_;
    }

    void addNewCorrelation(const float* xtx, float xty, int shift = 0) {
        const int corPos = filledSize_ + shift;

        int pos = corPos * (corPos + 1) / 2;
        for (int i = 0; i <= corPos; ++i) {
            XTX_[pos + i] += xtx[i];
        }
        XTy_[corPos] += xty;
        trace_ += xtx[corPos];
        maxUpdatedPos_ = std::max(maxUpdatedPos_, corPos + 1);
    }

    void addFullCorrelation(const float* x, float y, float weight) {
        for (int i = 0; i < filledSize_; ++i) {
            XTy_[i] += x[i] * y * weight;
        }

        int pos = 0;
        for (int i = 0; i < filledSize_; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                XTX_[pos + j] += x[i] * x[j] * weight;
            }
            pos += i + 1;
        }

        *weight_ += weight;
    }

    void fillXTX(Eigen::MatrixXf& XTX) const {
        int basePos = 0;
        for (int i = 0; i < maxUpdatedPos_; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                XTX(i, j) = XTX_[basePos + j];
                XTX(j, i) = XTX_[basePos + j];
            }
            basePos += i + 1;
        }
    }

    [[nodiscard]] Eigen::MatrixXf getXTX() const {
        Eigen::MatrixXf res(maxUpdatedPos_, maxUpdatedPos_);
        fillXTX(res);
        return res;
    }

    void fillXTy(Eigen::MatrixXf& XTy) const {
        for (int i = 0; i < maxUpdatedPos_; ++i) {
            XTy(i, 0) = XTy_[i];
        }
    }

    [[nodiscard]] Eigen::MatrixXf getXTy() const {
        Eigen::MatrixXf res(maxUpdatedPos_, 1);
        fillXTy(res);
        return res;
    }

    float getWeight() {
        return *weight_;
    }

    double getTrace() {
        return trace_;
    }

    // This one DOES NOT add up new correlations
    BinStat& operator+=(const BinStat& s) {
        *weight_ += *s.weight_;
        trace_ += s.trace_;

        int size = std::min(filledSize_, s.filledSize_);

        int pos = 0;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                XTX_[pos + j] += s.XTX_[pos + j];
            }
            pos += i + 1;
            XTy_[i] += s.XTy_[i];
        }
    }

    // This one DOES NOT subtract new correlations
    BinStat& operator-=(const BinStat& s) {
        *weight_ -= *s.weight_;
        trace_ -= s.trace_;

        int size = std::min(filledSize_, s.filledSize_);

        int pos = 0;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                XTX_[pos + j] -= s.XTX_[pos + j];
            }
            pos += i + 1;
            XTy_[i] -= s.XTy_[i];
        }
    }

private:
    friend BinStat operator+(const BinStat& lhs, const BinStat& rhs);
    friend BinStat operator-(const BinStat& lhs, const BinStat& rhs);

private:
    int size_;
    int filledSize_;
    int maxUpdatedPos_;

    float* XTX_;
    float* XTy_;
    float* weight_;

    double trace_;
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
