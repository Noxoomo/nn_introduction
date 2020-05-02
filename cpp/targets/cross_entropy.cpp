#include "cross_entropy.h"
#include <vec_tools/transform.h>
#include <vec_tools/stats.h>


inline void crossEntropyGradient(const Vec& target, const Vec& point, Vec to) {
    Vec p = VecTools::sigmoidCopy(point);
    VecTools::copyTo(target, to);
    to -= p;
}

void CrossEntropy::subsetDer(const Vec& point, const Buffer<int32_t>& indices, Vec to) const {
    Vec gatheredPoint(indices.size());
    Vec gatheredTarget(indices.size());
    VecTools::gather(point, indices, gatheredPoint);
    VecTools::gather(target_, indices, gatheredTarget);
    crossEntropyGradient(gatheredTarget, gatheredPoint, to);
}

Vec CrossEntropy::gradientTo(const Vec& x, Vec to) const {
    crossEntropyGradient(target_, x, to);
    return to;
}


DoubleRef CrossEntropy::valueTo(const Vec& x, DoubleRef to) const {
    auto tmp = VecTools::expCopy(x);
    tmp += 1;
    VecTools::log(tmp);
//
//    // t * log(p(x)) + (1.0 - t) * log(1.0 - p(x));
//    // t log(s(x)) + (1.0 - t) * log(1.0 - s(x))
//    //t log(s(x)) + (1.0 - t)  * log(s(-x))
//    //t (x - log(1 + exp(x)) + (1.0 - t) * (-log(1.0 + exp(x))
//    //t * x - log(1.0 + exp(x))
    double scoresSum = VecTools::sum(target_ * x - tmp);
    to = (scoresSum / x.dim());
    return to;
}

CrossEntropy::CrossEntropy(const DataSet& ds,
                           const Vec& target)
    : Stub(ds)
    , target_(target)  {

}
CrossEntropy::CrossEntropy(const DataSet& ds, double border)
    : Stub<Target, CrossEntropy>(ds)
    , target_(ds.samplesCount()) {
    auto targetAccessor = target_.arrayRef();
    auto sourceAccessor = ds.target().arrayRef();
    for (uint64_t i  = 0; i < targetAccessor.size(); ++i) {
        targetAccessor[i] = sourceAccessor[i] > border;
    }
}


CrossEntropy::CrossEntropy(const DataSet& ds)
    : Stub<Target, CrossEntropy>(ds)
      , target_(ds.target()) {
}
