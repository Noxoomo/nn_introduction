#pragma once

#include <vector>
#include <algorithm>


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

    multi_dim_array_idxs copyFrom(int from) {
        std::vector<int> newSizes(sizes_.begin() + from, sizes_.end());
        return multi_dim_array_idxs(newSizes);
    }
};

template <int N, typename T>
class MultiDimArray {
public:
    template <typename ... Args>
    explicit MultiDimArray(std::vector<int> sizes, Args... args)
            : idxs_(new multi_dim_array_idxs(std::move(sizes))) {
        shift_pos_ = 0;
        int size = idxs_->nElem(0);
        void* rawData = operator new[](size * sizeof(T));
        data_ = static_cast<T*>(rawData);
        for (int i = 0; i < size; ++i) {
            // TODO can't use std::forward for now since it won't work with copy constructors
            new(data_ + i) T(args...);
        }
    }

    MultiDimArray(T* data, multi_dim_array_idxs* idxs, int shift_pos)
            : data_(data)
            , idxs_(idxs)
            , shift_pos_(shift_pos) {

    }

    MultiDimArray() {
        data_ = nullptr;
        idxs_ = nullptr;
        shift_pos_ = 0;
    }

    MultiDimArray<N - 1, T> operator[](int idx) const {
        return MultiDimArray<N - 1, T>(data_ + idxs_->shifts_[shift_pos_] * idx, idxs_, shift_pos_ + 1);
    }

    MultiDimArray<N, T>& operator=(MultiDimArray<N, T>&& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(idxs_, other.idxs_);
        shift_pos_ = other.shift_pos_;
    }

    MultiDimArray(const MultiDimArray<N, T>& other) = delete;
    MultiDimArray(MultiDimArray<N, T>&& other) noexcept = default;

    MultiDimArray<N, T> copy() const {
        int size = idxs_->nElem(shift_pos_);
        void* rawData = operator new[](size * sizeof(T));
        auto newData = static_cast<T*>(rawData);
        for (int i = 0; i < size; ++i) {
            new(newData + i) T(data_[i]);
        }
        auto* newIdxs = new multi_dim_array_idxs(idxs_->copyFrom(shift_pos_));
        return MultiDimArray<N, T>(newData, newIdxs, 0);
    }

    T* data() const {
        return data_;
    }

    int size() const {
        return idxs_->nElem(shift_pos_);
    }

    ~MultiDimArray() {
        if (shift_pos_ == 0) {
            delete data_;
            delete idxs_;
        }
    }

private:
    T* data_;
    multi_dim_array_idxs* idxs_;
    int shift_pos_;
};

template <typename T>
class MultiDimArray<1, T> {
public:
    template <typename ... Args>
    explicit MultiDimArray(std::vector<int> sizes, Args... args)
            : idxs_(new multi_dim_array_idxs(std::move(sizes))) {
        shift_pos_ = 0;
        int size = idxs_->nElem(0);
        void* rawData = operator new[](size * sizeof(T));
        data_ = static_cast<T*>(rawData);
        for (int i = 0; i < size; ++i) {
            // TODO can't use std::forward for now since it won't work with copy constructors
            new(data_ + i) T(args...);
        }
    }

    MultiDimArray(T* data, multi_dim_array_idxs* idxs, int shift_pos)
            : data_(data)
            , idxs_(idxs)
            , shift_pos_(shift_pos) {

    }

    MultiDimArray(const MultiDimArray<1, T>& other) = delete;
    MultiDimArray(MultiDimArray<1, T>&& other) noexcept = default;

    T& operator[](int idx) {
        return data_[idx];
    }

    MultiDimArray<1, T>& operator=(MultiDimArray<1, T>&& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(idxs_, other.idxs_);
        shift_pos_ = other.shift_pos_;
    }

    T* data() const {
        return data_;
    }

    int size() const {
        return idxs_->nElem(shift_pos_);
    }

    MultiDimArray<1, T> copy() const {
        int size = idxs_->nElem(shift_pos_);
        void* rawData = operator new[](size * sizeof(T));
        auto newData = static_cast<T*>(rawData);
        for (int i = 0; i < size; ++i) {
            new(newData + i) T(data_[i]);
        }
        auto* newIdxs = new multi_dim_array_idxs(idxs_->copyFrom(shift_pos_));
        return MultiDimArray<1, T>(newData, newIdxs, 0);
    }

    ~MultiDimArray() {
        if (shift_pos_ == 0) {
            delete data_;
            delete idxs_;
        }
    }

private:
    T* data_;
    multi_dim_array_idxs* idxs_;
    int shift_pos_;
};
