#pragma once

#include <vector>


struct multi_dim_array_idxs {
    std::vector<int> sizes_;
    std::vector<int> shifts_;

    multi_dim_array_idxs() = default;

    explicit multi_dim_array_idxs(std::vector<int> sizes)
            : sizes_(std::move(sizes)) {
        shifts_.resize(sizes_.size(), 0);

        int shift = 1;
        for (int i = (int)sizes_.size() - 1; i >= 0; --i) {
            shifts_[i] = shift;
            shift *= sizes_[i];
        }
    }

    int nElem(int pos) const {
        return shifts_[pos] * sizes_[pos];
    }
};

template <int N, typename T>
class MultiDimArray {
public:
    MultiDimArray()
            : data_(nullptr)
            , idxs_(nullptr)
            , shift_pos_(0) {

    }

    MultiDimArray(T* data, multi_dim_array_idxs* idxs, int shift_pos)
            : data_(data)
            , idxs_(idxs)
            , shift_pos_(shift_pos) {

    }

    T* operator()(int idx1, int idx2, int idx3) {
        auto data1 = data_ + idxs_->shifts_[shift_pos_] * idx1;
        auto data2 = data1 + idxs_->shifts_[shift_pos_ + 1] * idx2;
        auto data3 = data2 + idxs_->shifts_[shift_pos_ + 2] * idx3;
        return data3;
//        return MultiDimArray<N - 1, T>(data_ + idxs_->shifts_[shift_pos_] * idx, idxs_, shift_pos_ + 1);
    }

    MultiDimArray<N - 1, T> operator[](int idx) {
        return MultiDimArray<N - 1, T>(data_ + idxs_->shifts_[shift_pos_] * idx, idxs_, shift_pos_ + 1);
    }

    T* data() const {
        return data_;
    }

    int nElem() const {
        return idxs_->nElem(shift_pos_);
    }

private:
    T* data_;
    multi_dim_array_idxs* idxs_;
    int shift_pos_;
};

template <typename T>
class MultiDimArray<1, T> {
public:
    MultiDimArray() = default;

    MultiDimArray(T* data, multi_dim_array_idxs* idxs, int shift_pos_)
            : data_(data) {

    }

    T& operator[](int idx) {
        return data_[idx];
    }

    T* data() const {
        return data_;
    }

private:
    T* data_;
};
