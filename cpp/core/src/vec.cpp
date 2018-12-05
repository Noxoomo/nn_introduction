#include "vec_impls.h"
#include <core/vec.h>

using namespace Impl;

#if defined(CUDA)

static void SetValue(CudaVec& vec, int64_t idx, double value) {
    float val = value;
    Cuda::CopyMemory(&val, vec.data() + idx, 1);;
}

static double GetValue(const CudaVec& vec, int64_t idx) {
    //don't do this in production, just for test
    float val;
    Cuda::CopyMemory(vec.data() + idx, &val, 1);
    return val;
}

#endif

template <class T>
static inline void SetValue(T& vec, int64_t idx, double value) {
    vec.set(idx, value);
}

template <class T>
static inline double GetValue(const T& vec, int64_t idx) {
    return vec.get(idx);
}

void Vec::set(int64_t index, double value) {
    assert(!immutable_);

    std::visit([&](auto instance) {
        SetValue(*instance, index, value);
    }, DynamicDispatch(anyVec()));
}

double Vec::get(int64_t index) const {
    return std::visit([&](auto instance) -> double {
        return GetValue(*instance, index);
    }, DynamicDispatch(anyVec()));
}

int64_t Vec::dim() const {
    auto vecVariant = DynamicDispatch(anyVec());
    return std::visit([&](auto instance) -> int64_t {
        return instance->dim();
    }, vecVariant);
}

//
//Vec::Vec(int64_t dim)
//: Vec(std::make_shared<PlaceholderVec>(dim)) {
//
//}


//TODO: should be placeholder
Vec::Vec(int64_t dim)
    : Vec(std::make_shared<ArrayVec>(dim)) {

}


Vec Vec::slice(int64_t from, int64_t size) {
    auto subvec = std::make_shared<SubVec>(*this,
                                           SliceIndexTransformation(from, size));
    return Vec(std::move(subvec));
}


Vec Vec::slice(int64_t from, int64_t size) const {
    auto subvec = std::make_shared<SubVec>(*this,
                                           SliceIndexTransformation(from, size));
    return Vec(std::move(subvec), true);
}


Vec VecRef::slice(int64_t from, int64_t size) {
    return asVecRef().slice(from, size);
}
Vec ConstVecRef::slice(int64_t from, int64_t size) const {
    return ptr_->slice(from, size);

}
