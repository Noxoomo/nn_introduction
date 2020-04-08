#pragma once

#include "target.h"
#include "stat_based_loss.h"
#include "linear_l2_stat.h"

#include <core/vec_factory.h>
#include <core/vec.h>
#include <vec_tools/fill.h>
#include <vec_tools/transform.h>
#include <vec_tools/distance.h>
#include <vec_tools/stats.h>
#include <util/parallel_executor.h>
#include <core/vec_factory.h>
#include <vec_tools/fill.h>


class RegularizedLinearL2Score {
public:
    explicit RegularizedLinearL2Score(double lambda = 0)
            : lambda_(lambda) {
        assert(lambda_ >= 0);
        lambda_ += 1e-10;
    }

    [[nodiscard]] double bestIncrement(const LinearL2Stat& comb) const {
        throw std::runtime_error("Unimplemented");
    }

    [[nodiscard]] double score(const LinearL2Stat& s) const {
        LinearL2Stat::EMx XTX = s.getXTX(0, 1);
        LinearL2Stat::EMx XTy = s.getXTy(1);
        LinearL2Stat::EMx wHat = s.getWHat(lambda_, 1);

        LinearL2Stat::EMx c1 = -2 * (XTy.transpose() * wHat);
        LinearL2Stat::EMx c2 = wHat.transpose() * XTX * wHat;
        LinearL2Stat::EMx reg = lambda_ * wHat.transpose() * wHat;
        LinearL2Stat::EMx res = c1 + c2 + reg;

        return res(0, 0);
    }

private:
    double lambda_ = 1;
};

class LinearL2LogScore {
public:
    explicit LinearL2LogScore(double lambda = 0)
            : lambda_(lambda) {
        assert(lambda_ >= 0);
        lambda_ += 1e-10;
    }

    [[nodiscard]] double bestIncrement(const LinearL2Stat& comb) const {
        throw std::runtime_error("Unimplemented");
    }

    [[nodiscard]] double score(const LinearL2Stat& s) const {
        const float weight = s.w_;
        if (weight < 2)
            return 0;
        LinearL2Stat::EMx wHat = s.getWHat(lambda_, 1); // n x 1 matrix
        LinearL2Stat::EMx xy = s.getXTy(1); // n x 1 matrix
        xy -= (s.sumY_ / weight) * s.getSumX(1);
        xy *= 1. / weight;
        float scoreFromLinear = weight * (wHat.transpose() * xy)(0, 0);
        float scoreFromConst = s.sumY_ * s.sumY_ / weight;
        float targetValue = scoreFromConst + scoreFromLinear - lambda_ * wHat.norm();
        return -targetValue;
    }

private:
    double lambda_ = 1;
};


//using LinearL2ScoreFunction = RegularizedLinearL2Score;
using LinearL2ScoreFunction = LinearL2LogScore;

class LinearL2 : public Stub<Target, LinearL2>,
                 public StatBasedLoss<LinearL2Stat>,
                 public PointwiseTarget {
public:
    LinearL2(const DataSet& ds, Vec target, LinearL2ScoreFunction scoreFunction = LinearL2ScoreFunction())
            : Stub<Target, LinearL2>(ds)
            , nzTargets_(std::move(target))
            , scoreFunction_(scoreFunction) {

    }

    explicit LinearL2(const DataSet& ds, LinearL2ScoreFunction scoreFunction = LinearL2ScoreFunction())
            : Stub<Target, LinearL2>(ds)
            , nzTargets_(ds.target())
            , scoreFunction_(scoreFunction) {

    }

    LinearL2(const DataSet& ds,
           Vec target,
           Vec weights,
           const Buffer<int32_t>& indices,
             LinearL2ScoreFunction scoreFunction = LinearL2ScoreFunction())
            :  Stub<Target, LinearL2>(ds)
            , nzTargets_(std::move(target))
            , nzWeights_(std::move(weights))
            , nzIndices_(indices)
            , scoreFunction_(scoreFunction) {

    }

    void makeStats(Buffer<LinearL2Stat>* stats, Buffer<int32_t>* indices) const override {
        throw std::runtime_error("Not implemented");
    }

    [[nodiscard]] double bestIncrement(const LinearL2Stat& comb) const override {
        throw std::runtime_error("Unimplemented");
    }

    [[nodiscard]] double score(const LinearL2Stat& comb) const override {
        return scoreFunction_.score(comb);
    }

    void subsetDer(const Vec& point, const Buffer<int32_t>& indices, Vec to) const override {
        assert(point.dim() == nzTargets_.dim());
        assert(indices.size() == to.dim());

        auto destArrayRef = to.arrayRef();
        auto targetArrayRef = nzTargets_.arrayRef();
        auto indicesArrayRef = indices.arrayRef();
        auto sourceArrayRef = point.arrayRef();

        for (int64_t i = 0; i < indices.size(); ++i) {
            const int32_t idx = indicesArrayRef[i];
            destArrayRef[i] = targetArrayRef[idx] - sourceArrayRef[idx];
        }
    }

    class Der : public Stub<Trans, Der> {
    public:
        Der(const LinearL2& owner)
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
        const LinearL2& owner_;
    };

    DoubleRef valueTo(const Vec& x, DoubleRef to) const {
        assert(nzWeights_.dim() == 0);
        assert(nzIndices_.size() == 0);
        to = VecTools::sum((x - nzTargets_) ^ 2);
        to /= x.dim();
        to = sqrt(to);
        return to;
    }

    [[nodiscard]] Vec targets() const override {
        if (nzIndices_.size() == 0  && nzTargets_.size() != 0) {
            std::cout << "full targets" << std::endl;
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

    [[nodiscard]] Vec weights() const override {
        auto weights = VecFactory::create(ComputeDeviceType::Cpu, ds_.samplesCount());

        if (nzIndices_.size() == 0) {
            if (nzTargets_.size() != 0) {
                VecTools::fill(1.0, weights);
                std::cout << "uniform weights" << std::endl;
                return weights;
            } else {
                VecTools::fill(0.0, weights);
                std::cout << "no weights" << std::endl;
                return weights;
            }
        }

        std::cout << "partial weights" << std::endl;

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
    Vec nzTargets_;
    Vec nzWeights_;
    Buffer<int32_t> nzIndices_;
    LinearL2ScoreFunction scoreFunction_;
};
