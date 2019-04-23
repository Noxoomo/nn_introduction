#pragma once

#include "vec.h"
#include "scalar.h"

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
