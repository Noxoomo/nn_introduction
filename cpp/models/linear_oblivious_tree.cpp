#include "linear_oblivious_tree.h"


void LinearObliviousTree::applyToBds(const BinarizedDataSet& bds, Mx to, ApplyType type) const {
    const auto &ds = bds.owner();
    const uint64_t sampleDim = ds.featuresCount();
    const uint64_t targetDim = to.xdim();

    ConstVecRef<float> dsRef = ds.samplesMx().arrayRef();
    VecRef<float> toRef = to.arrayRef();

    uint64_t xSliceStart = 0;
    uint64_t toSliceStart = 0;

    for (int64_t i = 0; i < ds.samplesCount(); ++i) {
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

    for (uint32_t i = 0; i < splits_.size(); ++i) {
        const auto& s = splits_[i];
        auto origFId = std::get<0>(s);
        auto border = std::get<1>(s);

        const auto val = x[origFId];
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
    leaves_[lId].grad(toRef, scale_);
}

void LinearObliviousTree::printInfo() const {
    std::cout << leaves_.size() << "->(";
    for (int i = 0; i < (int)splits_.size(); ++i) {
        int origFId = std::get<0>(splits_[i]);
        double border = std::get<1>(splits_[i]);
        std::cout << "f[" << origFId << "] > " << border;
        if (i != (int)splits_.size() - 1) {
            std::cout << ", ";
        } else {
            std::cout << ")+";
        }
    }

    std::cout << "[";
    for (int lId = 0; lId < (int)leaves_.size(); ++lId) {
        leaves_[lId].printInfo(scale_);
        if (lId != (int)leaves_.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}
