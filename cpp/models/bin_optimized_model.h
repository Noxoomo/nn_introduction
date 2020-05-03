#pragma once

#include "model.h"
#include <data/dataset.h>
#include <data/binarized_dataset.h>


class BinOptimizedModel : public virtual Model {
public:
    using Model::apply;
    using Model::append;

    void apply(const BinarizedDataSet& ds, Mx to) const {
        applyToBds(ds, to, ApplyType::Set);
    }

    void append(const BinarizedDataSet& ds, Mx to) const {
        applyToBds(ds, to, ApplyType::Append);
    }

    virtual GridPtr gridPtr() const = 0;

    operator std::unique_ptr<BinOptimizedModel>() const {
        return cloneBinModelUnique(1.0);
    }

    operator std::shared_ptr<BinOptimizedModel>() const {
        return cloneBinModelShared(1.0);
    }

protected:


    virtual void applyToBds(const BinarizedDataSet& ds, Mx to, ApplyType type) const = 0;

    virtual void applyToDs(const DataSet& ds, Mx to) const {
        if (gridPtr()) {
            // TODO cachedBinarize... hardcode sizes for now
            const auto& bds = cachedBinarize(ds, gridPtr(), gridPtr()->origFeaturesCount());
            applyToBds(bds, to, ApplyType::Set);
        } else {
            Model::applyToDs(ds, to);
        }
    }

    virtual void appendToDs(const DataSet& ds, Mx to) const {
        if (gridPtr()) {
            // TODO cachedBinarize... hardcode sizes for now
            const auto& bds = cachedBinarize(ds, gridPtr(), gridPtr()->origFeaturesCount());
            applyToBds(bds, to, ApplyType::Append);
        } else {
            Model::appendToDs(ds, to);
        }
    }

protected:
    virtual std::unique_ptr<BinOptimizedModel> cloneBinModelUnique(double scale) const  = 0;

    virtual std::shared_ptr<BinOptimizedModel> cloneBinModelShared(double scale) const  = 0;

    virtual std::unique_ptr<Model> cloneModelUnique(double scale) const {
        return std::unique_ptr<BinOptimizedModel>(cloneBinModelUnique(scale).release());
    }

    virtual std::shared_ptr<Model> cloneModelShared(double scale) const {
        return std::static_pointer_cast<Model>(cloneBinModelShared(scale));
    }
};



template <class Impl>
class Stub<BinOptimizedModel, Impl> : public virtual BinOptimizedModel, public Stub<Trans, Impl> {
public:

    Stub(int64_t xdim, int64_t ydim)
    : Stub<Trans, Impl>(xdim, ydim) {

    }

    Stub(const Impl& other)
    : Stub<Trans, Impl>(other) {

    }

    Vec trans(const Vec& x, Vec to) const override {
        Buffer<uint8_t> bins = Buffer<uint8_t>::create(x.dim());
        static_cast<const Impl*>(this)->grid().binarize(x, bins);
        static_cast<const Impl*>(this)->applyBinarizedRow(bins, to);
        return to;
    }

protected:
    std::unique_ptr<BinOptimizedModel> cloneBinModelUnique(double scale) const override {
        return std::unique_ptr<BinOptimizedModel>(new Impl(*static_cast<const Impl*>(this), scale));
    }

    std::shared_ptr<BinOptimizedModel> cloneBinModelShared(double scale) const override {
        return std::static_pointer_cast<BinOptimizedModel>(std::make_shared<Impl>(*static_cast<const Impl*>(this), scale));
    }
};
