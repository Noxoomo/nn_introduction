#include "vec.h"
#include "vec_factory.h"
#include "torch_helpers.h"
#include "matrix.h"


void Vec::set(int64_t index, double value) {
    data().accessor<float, 1>()[index] = value;
}

double Vec::get(int64_t index) const {
    return data().accessor<float, 1>()[index];
}

int64_t Vec::dim() const {
    return TorchHelpers::totalSize(data());
}

Vec Vec::append(float val) const {
    auto data = this->data();
    auto valT = torch::ones({1}, torch::kFloat);
    valT *= val;
    auto newData = torch::cat({data, valT});
    auto res = Vec(0);
    res.data_ = newData;
    return res;
}

//
////TODO: should be placeholder
Vec::Vec(int64_t dim, float value)
    : Buffer<float>(torch::ones({dim}, TorchHelpers::tensorOptionsOnDevice(CurrentDevice())) * value) {

}

Vec::Vec(int64_t dim, const ComputeDevice& device)
    : Buffer<float>(torch::zeros({dim}, TorchHelpers::tensorOptionsOnDevice(device))) {
}

Vec Vec::slice(int64_t from, int64_t size) {
    assert(data().dim() == 1);
    return Vec(data().slice(0, from, from + size));
}

Vec Vec::slice(int64_t from, int64_t size) const {
    return Vec(data().slice(0, from, from + size));
}

Vec& Vec::operator+=(const Vec& other) {
    data() += other;
    return *this;
}
Vec& Vec::operator-=(const Vec& other) {
    data() -= other;
    return *this;
}
Vec& Vec::operator*=(const Vec& other) {
    data() *= other;
    return *this;
}
Vec& Vec::operator/=(const Vec& other) {
    data() /= other;
    return *this;
}
Vec& Vec::operator+=(Scalar value) {
    data() += value;
    return *this;
}
Vec& Vec::operator-=(Scalar value) {
    data() -= value;
    return *this;
}

Vec& Vec::operator*=(Scalar value) {
    data() *= value;
    return *this;
}
Vec& Vec::operator/=(Scalar value) {
    data() /= value;
    return *this;
}
Vec& Vec::operator^=(const Vec& other) {
    data().pow_(other);
    return *this;
}
Vec& Vec::operator^=(Scalar q) {
    data().pow_(q);
    return *this;
}



Vec operator+(const Vec& left, const Vec& right) {
    auto result = VecFactory::uninitializedCopy(left);
    at::add_out(result, left, right);
    return result;
}
Vec operator-(const Vec& left, const Vec& right) {
    auto result = VecFactory::uninitializedCopy(left);
    at::sub_out(result, left, right);
    return result;
}
Vec operator*(const Vec& left, const Vec& right) {
    auto result = VecFactory::uninitializedCopy(left);
    at::mul_out(result, left, right);
    return result;
}
Vec operator/(const Vec& left, const Vec& right) {
    auto result = VecFactory::uninitializedCopy(left);
    at::div_out(result, left, right);
    return result;
}
Vec operator^(const Vec& left, Scalar q) {
    auto result = VecFactory::uninitializedCopy(left);
    at::pow_out(result, left, q);
    return result;
}

Vec operator+(const Vec& left, Scalar right) {
    auto result = VecFactory::clone(left);
    result += right;
    return result;
}

Vec operator-(const Vec& left, Scalar right) {
    auto result = VecFactory::clone(left);
    result -= right;
    return result;
}
Vec operator*(const Vec& left, Scalar right) {
    auto result = VecFactory::clone(left);
    result *= right;
    return result;
}

Vec operator/(const Vec& left, Scalar right) {
    auto result = VecFactory::clone(left);
    result /= right;
    return result;
}
Vec operator^(const Vec& left, const Vec& right) {
    auto result = VecFactory::clone(left);
    result ^= right;
    return result;
}
Vec operator>(const Vec& left, Scalar right) {
    return Vec(left.data().gt(right).to(torch::ScalarType::Float));
}
Vec operator<(const Vec& left, Scalar right) {
    return Vec(left.data().lt(right).to(torch::ScalarType::Float));
}
Vec eq(const Vec& left, Scalar right) {
    return Vec(left.data().eq(right).to(torch::ScalarType::Float));
}
Vec eq(const Vec& left, const Vec& right) {
    return Vec(left.data().eq(right).to(torch::ScalarType::Float));
}
Vec operator!=(const Vec& left, Scalar right) {
    return Vec(left.data().ne(right).to(torch::ScalarType::Float));
}

double l2(const Vec& x) {
    double res = 0;
    auto xRef = x.arrayRef();
    for (float i : xRef) {
        res += i * i;
    }
    return res;
}
