#pragma once

#include "vec.h"
#include "scalar.h"

#include <iostream>

enum class MatrixLayout {
    RowMajor
};

class Mx  : public Vec {
public:
    Mx(Mx&& other) = default;
    Mx(Mx& other) = default;

    Mx(Vec& x, int64_t nrows, int64_t ncols, MatrixLayout layout = MatrixLayout::RowMajor)
        : Vec(x)
          , layout_(layout)
          , rows_(nrows)
          , cols_(ncols) {
    }

    Mx(const Vec& x,
        int64_t nrows,
        int64_t ncols,
        MatrixLayout layout = MatrixLayout::RowMajor
    )
        : Vec(x)
          , layout_(layout)
          , rows_(nrows)
          , cols_(ncols) {
        assert(dim() == rows_ * cols_);
    }

    Mx(const Mx& other)
        : Mx(other, other.rows_, other.cols_, other.layout_) {

    }

    Mx(int64_t nrows,
       int64_t ncols,
       MatrixLayout layout = MatrixLayout::RowMajor)
        : Vec(nrows * ncols)
          , layout_(layout)
          , rows_(nrows)
          , cols_(ncols) {
        assert(dim() == rows_ * cols_);
    }

    Mx& set(int64_t x, int64_t y, Scalar val);

    Scalar get(int64_t x, int64_t y) const;

    Mx T() const {
        auto newData = this->data().view({ydim(), xdim()}).transpose(0, 1).contiguous().view({-1});
        auto tmpVec = Vec(newData);
        return Mx(tmpVec, xdim(), ydim());
    }

    Mx inverse() const {
        assert(rows_ == cols_);
        Mx res(*this);
//        auto tensor = torch::cholesky_inverse(torch::cholesky(this->data().view({rows_, cols_}))).contiguous().view({-1});
        auto tensor = torch::inverse(this->data().view({rows_, cols_})).contiguous().view({-1});
        return Mx(Vec(tensor), rows_, cols_);
    }

    Mx XXT() const;

    Mx& operator+=(const Mx& other);
    Mx& operator-=(const Mx& other);

    Mx& operator+=(Scalar value);
    Mx& operator-=(Scalar value);
    Mx& operator*=(Scalar value);
    Mx& operator/=(Scalar value);

    Mx& operator^=(Scalar q);

    Vec row(int64_t idx);
    Vec row(int64_t idx) const;


    template <class Visitor>
    void iterateOverColumn(int64_t columnIdx, Visitor&& visitor) const {
        ConstVecRef<float> data = arrayRef();
        for (int64_t i = 0; i < ydim(); ++i) {
            visitor(i, data[seqIndex(columnIdx, i)]);
        }
    }

    void addColumn(const Vec& column, bool last = false);

    int64_t xdim() const {
        return cols_;
    }

    int64_t ydim() const {
        return rows_;
    }
private:
    int64_t seqIndex(int64_t x, int64_t y) const;
private:
    MatrixLayout layout_ = MatrixLayout::RowMajor;
    int64_t rows_ = 0;
    int64_t cols_ = 0;
};

Mx operator*(const Mx& A, const Mx& B);
Mx operator*(const Mx& A, Scalar s);
Mx operator-(const Mx& A, const Mx& B);
Mx operator+(const Mx& A, const Mx& B);

inline std::ostream& operator<<(std::ostream& os, const Mx& m)
{
    os << "[";
    for (int i = 0; i < m.ydim(); ++i) {
        if (i != 0)
            os << " ";
        for (int j = 0; j < m.xdim(); ++j) {
            os << std::setw(5) << m.get(j, i) << " ";
        }
        os << "\n";
    }
    os << "]";
    return os;
}

inline Mx Diag(int n, double val = 1.0) {
    Mx res(n, n);
    for (int i = 0; i < n; ++i) {
        res.set(i, i, val);
    }
    return res;
}

inline Mx I(int n) {
    return Diag(n);
}
