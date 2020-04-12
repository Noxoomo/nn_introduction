#include "l2.h"

void L2::makeStats(Buffer<L2Stat>* stats, Buffer<int32_t>* indices) const {
    (*stats) = Buffer<L2Stat>(nzTargets_.dim());
    if (nzIndices_.size()) {
        (*indices) = nzIndices_.copy();
    } else {
        (*indices) = Buffer<int32_t>(nzTargets_.dim());
        auto indicesRef = indices->arrayRef();
        for (int32_t i = 0; i < indicesRef.size(); ++i) {
            indicesRef[i] = i;
        }
    }

    auto nzTargetsRef = nzTargets_.arrayRef();
    auto nzWeightsRef = nzWeights_.dim() ? nzWeights_.arrayRef() : ConstVecRef<float>((const float*)nullptr, (size_t)0u);

    VecRef<L2Stat> statsRef = stats->arrayRef();
    if (!nzWeightsRef.empty()) {
        parallelFor(0, nzTargetsRef.size(), [&](int64_t i) {
            statsRef[i].Sum = nzWeightsRef[i] * nzTargetsRef[i];
            statsRef[i].Weight = nzWeightsRef[i];
        });
    } else {
        parallelFor(0, nzTargetsRef.size(), [&](int64_t i) {
            statsRef[i].Sum = nzTargetsRef[i];
            statsRef[i].Weight = 1.0;
        });
    }
}

void L2::subsetDer(const Vec& point, const Buffer<int32_t>& indices, Vec to) const {
    //TODO(Noxoomo): support sparse target
    assert(point.dim() == nzTargets_.dim());
    assert(indices.size() == to.dim());

    auto destArrayRef = to.arrayRef();
    auto targetArrayRef = nzTargets_.arrayRef();
    auto indicesArrayRef = indices.arrayRef();
    auto sourceArrayRef = point.arrayRef();

    for (int64_t i = 0; i < indices.size(); ++i) {
        const int32_t idx = indicesArrayRef[i];
        destArrayRef[i] =  targetArrayRef[idx]  -  sourceArrayRef[idx];
    }
}
