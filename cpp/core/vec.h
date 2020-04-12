#pragma once

#include "context.h"
#include "scalar.h"
#include "buffer.h"
#include <torch/torch.h>
#include <cassert>
#include <cstdint>
#include <variant>
#include <memory>
#include <utility>
#include <functional>
#include <util/array_ref.h>

class Vec : public Buffer<float> {
public:
    explicit Vec(int64_t dim, float value = 0);

    Vec()
    : Vec(0) {
    }

    Vec(Vec&& other) = default;

    Vec(Vec& other) = default;

    Vec(const Vec& other)
        : Buffer<float>(other) {
    }

    explicit Vec(int64_t dim, const ComputeDevice& device);

    explicit Vec(torch::Tensor impl)
        : Buffer<float>(std::move(impl)) {
    }

    Vec append(float val) const;

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

    bool operator==(const Vec& other) const {
        return data().equal(other.data());
    }

protected:
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


Vec operator>(const Vec& left, Scalar right);
Vec operator<(const Vec& left, Scalar right);
Vec eq(const Vec& left, Scalar right);
Vec eq(const Vec& left, const Vec& right);
Vec operator!=(const Vec& left, Scalar right);

double l2(const Vec& x);