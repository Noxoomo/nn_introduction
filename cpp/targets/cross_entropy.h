#pragma once

#include "target.h"
#include <util/parallel_executor.h>
#include <core/vec_factory.h>
#include <vec_tools/fill.h>

class CrossEntropy :  public Stub<Target, CrossEntropy>,
                      public  PointwiseTarget {
public:

    CrossEntropy(const DataSet& ds,
                 const Vec& target_);

    explicit CrossEntropy(const DataSet& ds, double border);
    explicit CrossEntropy(const DataSet& ds);


    class Der : public Stub<Trans, Der> {
    public:
        Der(const CrossEntropy& owner)
           :  Stub<Trans, Der>(owner.dim(), owner.dim())
           , owner_(owner) {

        }

        Vec trans(const Vec& x, Vec to) const final {
            return owner_.gradientTo(x, to);
        }

    private:
        const CrossEntropy& owner_;
    };


    Vec gradientTo(const Vec& x, Vec to) const override;

    void subsetDer(const Vec& point, const Buffer<int32_t>& indices, Vec to) const override;

    DoubleRef valueTo(const Vec& x, DoubleRef to) const;

    Vec targets() const override {
        return target_;
    }

    Vec weights() const override {
        auto w = VecFactory::create(ComputeDeviceType::Cpu, target_.size());
        VecTools::fill(1.0, w);
        return w;
    }

private:
    Vec target_;
};
