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

void LinearObliviousTree::appendTo(const Vec& x, Vec to) const {
    to += value(x.arrayRef());
}

int LinearObliviousTree::getLeaf(const ConstVecRef<float>& x) const {
    int lId = 0;

    for (int i = 0; i < splits_.size(); ++i) {
        const auto& s = splits_[i];
        auto fId = std::get<0>(s);
        auto condId = std::get<1>(s);

        const auto border = grid_->condition(fId, condId);
        const auto val = x[grid_->origFeatureIndex(fId)];
        if (val > border) {
            lId |= 1 << (splits_.size() - i - 1);
        }
    }

    return lId;
}

double LinearObliviousTree::value(const ConstVecRef<float>& x) const {
    int lId = getLeaf(x);
    return scale_ * leaves_[lId].value(x);
}

double LinearObliviousTree::value(const Vec& x) {
    return value(x.arrayRef());
}

void LinearObliviousTree::grad(const Vec& x, Vec to) {
    VecRef<float> toRef = to.arrayRef();
    int lId = getLeaf(x.arrayRef());
    leaves_[lId].grad(toRef);
}

void LinearObliviousTree::printInfo() const {
    std::cout << leaves_.size() << "->(";
    for (int i = 0; i < (int)splits_.size(); ++i) {
        int fId = std::get<0>(splits_[i]);
        int condId = std::get<1>(splits_[i]);
        const auto border = grid_->condition(fId, condId);
        std::cout << "f[" << fId << "] > " << border;
        if (i != splits_.size() - 1) {
            std::cout << ", ";
        } else {
            std::cout << ")+";
        }
    }

    std::cout << "[";
    for (int lId = 0; lId < (int)leaves_.size(); ++lId) {
        leaves_[lId].printInfo();
        if (lId != leaves_.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}
