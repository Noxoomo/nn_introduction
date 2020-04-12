#pragma once

#include <torch/torch.h>
#include <cassert>
#include <cstdint>
#include <variant>
#include <memory>
#include <utility>
#include <functional>

class Scalar {
public:

    Scalar(const torch::Scalar& val)
        : value_(val) {

    }

    Scalar(double val)
        : value_(torch::Scalar(val)) {

    }

    Scalar(const torch::Tensor& tensor)
        : value_(tensor) {
        assert(tensor.dim() == 0);
    }

    operator torch::Scalar() const {
        return torch::Scalar(value());
    }

    operator double() const {
        return value();
    }

    double value() const {
        if (value_.index() == 0) {
            torch::Scalar result = std::get<0>(value_);
            return result.to<double>();
        } else {
            assert(value_.index() == 1);
            const torch::Tensor& result = std::get<1>(value_);
            return result.data_ptr<float>()[0];
        }
    }

private:
    std::variant<torch::Scalar, torch::Tensor> value_;
};


