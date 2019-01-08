#pragma once

#include "context.h"
#include "scalar.h"
#include <torch/torch.h>
#include <cassert>
#include <cstdint>
#include <variant>
#include <memory>
#include <utility>
#include <functional>

class Vec {
public:
    explicit Vec(int64_t dim);

    Vec(Vec&& other) = default;

    Vec(Vec& other) = default;

    Vec(const Vec& other)
        : vec_(other.vec_)
          , immutable_(true) {
    }

    void set(int64_t index, double value);

    Vec slice(int64_t from, int64_t size);

    Vec slice(int64_t from, int64_t size) const;

    double get(int64_t index) const;

    double operator()(int64_t index) const {
        return get(index);
    }

    Vec& operator+=(const Vec& other);
    Vec& operator-=(const Vec& other);
    Vec& operator*=(const Vec& other);
    Vec& operator/=(const Vec& other);

    Vec& operator+=(Scalar value);
    Vec& operator-=(Scalar value);
    Vec& operator*=(Scalar value);
    Vec& operator/=(Scalar value);

    Vec& operator^=(const Vec& other);
    Vec& operator^=(Scalar q);

    int64_t dim() const;

    bool immutable() const {
        return immutable_;
    }

    operator const torch::Tensor&() const {
        return data();
    }

    operator torch::Tensor&() {
        return data();
    }

    const torch::Tensor& data() const {
        return vec_;
    }

    torch::Tensor& data() {
        assert(!immutable_);
        return vec_;
    }

protected:

    explicit Vec(int64_t dim, const ComputeDevice& device);

    explicit Vec(
        const torch::Tensor& impl,
        bool immutable = false)
        : vec_(impl)
          , immutable_(immutable) {

    }

    torch::Tensor vec_;
    bool immutable_ = true;

    friend class VecFactory;
};

Vec operator+(const Vec& left, const Vec& right);
Vec operator-(const Vec& left, const Vec& right);
Vec operator*(const Vec& left, const Vec& right);
Vec operator/(const Vec& left, const Vec& right);

Vec operator+(const Vec& left, Scalar right);
Vec operator-(const Vec& left, Scalar right);
Vec operator*(const Vec& left, Scalar right);
Vec operator/(const Vec& left, Scalar right);

Vec operator^(const Vec& left, Scalar q);
Vec operator^(const Vec& left, const Vec& right);