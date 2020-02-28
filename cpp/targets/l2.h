#pragma once

#include "target.h"
#include "stat_based_loss.h"
#include <vec_tools/transform.h>
#include <vec_tools/distance.h>
#include <vec_tools/stats.h>
#include <util/parallel_executor.h>
#include <core/vec_factory.h>
#include <vec_tools/fill.h>

struct L2Stat {
    using Numeric = double;
    Numeric Sum = 0;
    Numeric Weight = 0;

    L2Stat& operator+=(const L2Stat& other) {
        Sum += other.Sum;
        Weight += other.Weight;
        return *this;
    }

    L2Stat& operator-=(const L2Stat& other) {
        Sum -= other.Sum;
        Weight -= other.Weight;
        return *this;
    }

    void clear() {
        Sum = Weight = 0;
    }
};

inline L2Stat operator-(const L2Stat& left, const L2Stat& right) {
    L2Stat res = left;
    res -= right;
    return res;
}

inline L2Stat operator+(const L2Stat& left, const L2Stat& right) {
    L2Stat res = left;
    res += right;
    return res;
}



class RegularizedL2Score {
public:
    explicit RegularizedL2Score(double lambda = 0)
    : lambda_(lambda) {
        assert(lambda_ >= 0);
        lambda_ += 1e-20;
    }

    double bestIncrement(const L2Stat& stat) const {
        return stat.Weight > 1e-20 ? stat.Sum / (stat.Weight + lambda_) : 0;
    }

    double score(const L2Stat& stat) const {
        return stat.Weight > 1e-20 ? -stat.Sum * stat.Sum / (stat.Weight + lambda_) : 0;
    }
private:
    double lambda_ = 1;
};

class LogL2Score {
public:
    double bestIncrement(const L2Stat& stat) const {
        return stat.Weight > 1 ? stat.Sum / (stat.Weight + 1e-20) : 0;
    }

    double score(const L2Stat& stat) const {
        return stat.Weight > 1 ? -stat.Sum * stat.Sum * (1. + 2. * log(stat.Weight + 1)) / stat.Weight : 0;
    }
};

//using ScoreFunction = RegularizedL2Score;
using ScoreFunction = LogL2Score;

class L2 :  public Stub<Target, L2>,
            public StatBasedLoss<L2Stat>,
            public PointwiseTarget {
public:

    explicit L2(const DataSet& ds, Vec target, ScoreFunction scoreFunction = ScoreFunction())
    : Stub<Target, L2>(ds)
    , nzTargets_(target)
    , scoreFunction_(scoreFunction) {

    }

    explicit L2(const DataSet& ds, ScoreFunction scoreFunction = ScoreFunction())
        : Stub<Target, L2>(ds)
          , nzTargets_(ds.target())
          , scoreFunction_(scoreFunction) {

    }


    L2(const DataSet& ds,
       Vec target,
       Vec weights,
       const Buffer<int32_t>& indices,
       ScoreFunction scoreFunction = ScoreFunction()
       )
        :  Stub<Target, L2>(ds)
        , nzTargets_(std::move(target))
        , nzWeights_(std::move(weights))
        , nzIndices_(indices)
        , scoreFunction_(scoreFunction) {

    }


    class Der : public Stub<Trans, Der> {
    public:
        Der(const L2& owner)
            :  Stub<Trans, Der>(owner.dim(), owner.dim())
               , owner_(owner) {

        }

        Vec trans(const Vec& x, Vec to) const final {
            //TODO(noxoomo): support subsets
             assert(x.dim() == owner_.nzTargets_.dim());

            VecTools::copyTo(owner_.nzTargets_, to);
            to -= x;
            return to;
        }

    private:
        const L2& owner_;
    };

    double bestIncrement(const L2Stat& stat) const  override {
        return scoreFunction_.bestIncrement(stat);
    }


    void makeStats(Buffer<L2Stat>* stats, Buffer<int32_t>* indices) const override;

    double score(const L2Stat& comb) const override {
        return scoreFunction_.score(comb);
    }

    void subsetDer(const Vec& point, const Buffer<int32_t>& indices, Vec to) const override;

    DoubleRef valueTo(const Vec& x, DoubleRef to) const {
        assert(nzWeights_.dim() == 0);
        assert(nzIndices_.size() == 0);
        to = VecTools::sum((x - nzTargets_) ^ 2);
        to /= x.dim();
        to = sqrt(to);
        return to;
    }

    Vec targets() const override {
        if (nzIndices_.size() == 0) {
            return nzTargets_;
        }

        auto targets = VecFactory::create(ComputeDeviceType::Cpu, ds_.samplesCount());
        auto tRef = targets.arrayRef();
        auto indicesRef = nzIndices_.arrayRef();
        auto nzTRef = nzTargets_.arrayRef();

        for (int64_t i = 0; i < nzIndices_.size(); ++i) {
            auto idx = indicesRef[i];
            tRef[idx] += nzTRef[i];
        }

        return targets;
    }

    Vec weights() const override {
        auto weights = VecFactory::create(ComputeDeviceType::Cpu, ds_.samplesCount());

        if (nzIndices_.size() == 0) {
            VecTools::fill(1.0, weights);
            return weights;
        }

        auto wRef = weights.arrayRef();
        auto indicesRef = nzIndices_.arrayRef();
        auto nzWRef = nzWeights_.arrayRef();

        for (int64_t i = 0; i < nzIndices_.size(); ++i) {
            auto idx = indicesRef[i];
            wRef[idx] += nzWRef[i];
        }

        return weights;
    }

private:
    //TODO(noxoomo): this should be just sparse vec instead
    Vec nzTargets_;
    Vec nzWeights_;
    Buffer<int32_t> nzIndices_;
    ScoreFunction scoreFunction_;

};


