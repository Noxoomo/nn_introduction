#include "linear_oblivious_tree.h"


void LinearObliviousTree::applyToBds(const BinarizedDataSet& bds, Mx to, ApplyType type) const {
    const auto &ds = bds.owner();
    const uint64_t sampleDim = ds.featuresCount();
    const uint64_t targetDim = to.xdim();

    ConstVecRef<float> dsRef = ds.samplesMx().arrayRef();
    VecRef<float> toRef = to.arrayRef();

    uint64_t xSliceStart = 0;
    uint64_t toSliceStart = 0;

    for (uint64_t i = 0; i < ds.samplesCount(); ++i) {
        ConstVecRef<float> x = dsRef.slice(xSliceStart, sampleDim);
        VecRef<float> y = toRef.slice(toSliceStart, targetDim);

        switch (type) {
            case ApplyType::Append:
                y[0] += value(x);
                break;
            case ApplyType::Set:
            default:
                y[0] = value(x);
        }

        xSliceStart += sampleDim;
        toSliceStart += targetDim;
    }
}

void LinearObliviousTree::appendTo(const Vec &x, Vec to) const {
    to += value(x.arrayRef());
}

double LinearObliviousTree::value(const ConstVecRef<float>& x) const {
    unsigned int lId = 0;

    for (int i = 0; i < splits_.size(); ++i) {
        const auto &s = splits_[i];
        auto fId = std::get<0>(s);
        auto condId = std::get<1>(s);

        const auto border = grid_->condition(fId, condId);
        const auto val = x[grid_->origFeatureIndex(fId)];
        if (val > border) {
            lId |= 1U << (splits_.size() - i - 1);
        }
    }

    return scale_ * leaves_[lId].value(x);
}

double LinearObliviousTree::value(const Vec &x) {
    return value(x.arrayRef());
}

void LinearObliviousTree::grad(const Vec &x, Vec to) {
    throw std::runtime_error("Unimplemented");
}
