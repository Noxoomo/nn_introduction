#include "oblivious_tree.h"


void ObliviousTree::applyBinarizedRow(const Buffer<uint8_t>& x, Vec to) const {
    assert(x.device().deviceType() == ComputeDeviceType::Cpu);
    assert(to.device().deviceType() == ComputeDeviceType::Cpu);
    assert(to.dim() == 1);
    auto bytes = x.arrayRef();

    int32_t bin = 0;
    for (int64_t f = 0; f < splits_.size(); ++f) {
        if (bytes[splits_[f].featureId_] > splits_[f].conditionId_) {
            bin |= 1 << f;
        }
    }

    to.set(0, static_cast<float>(leaves_.get(bin)));
}

void ObliviousTree::applyToBds(const BinarizedDataSet& ds, Mx to, ApplyType type) const {
    assert(to.ydim() == ds.samplesCount());
    auto bins = Buffer<uint32_t>::create(ds.samplesCount());
    bins.fill(0);
    auto binsArray = bins.arrayRef();

    for (int64_t i = 0; i < splits_.size(); ++i) {
        auto binFeature = splits_[i];
        //TODO:: this is map operation (but for non-trivial accessor)
        ds.visitFeature(binFeature.featureId_, [&](const int64_t lineIdx, int32_t bin) {
            if (bin > binFeature.conditionId_) {
                binsArray[lineIdx] |= (1 << i);
            }
        }, true);
    }


    VecRef<float> dstArray = static_cast<Vec>(to).arrayRef();
    ConstVecRef<float> leavesRef = leaves_.arrayRef();

    if (type == ApplyType::Set) {
        //TODO(noxoomo): this is gather primitive
        parallelFor(0, binsArray.size(), [&](int64_t i) {
            dstArray[i] = leavesRef[binsArray[i]];
        });
    } else {
        parallelFor(0, binsArray.size(), [&](int64_t i) {
            dstArray[i] += leavesRef[binsArray[i]];
        });
    }
}


void ObliviousTree::appendTo(const Vec& x, Vec to) const {
    assert(x.device().deviceType() == ComputeDeviceType::Cpu);
    assert(to.device().deviceType() == ComputeDeviceType::Cpu);
    assert(to.dim() == 1);


    int32_t bin = 0;
    for (int64_t f = 0; f < splits_.size(); ++f) {
        const auto binFeature = splits_[f];
        const auto border = grid_->condition(binFeature.featureId_, binFeature.conditionId_);
        const auto val = x.get(grid_->origFeatureIndex(binFeature.featureId_));
        if (val > border) {
            bin |= 1 << f;
        }
    }

    to += leaves_.get(bin);
}


double ObliviousTree::value(const Vec& x) {
    std::vector<double> probs(splits_.size());
    auto xRef = x.arrayRef();
    for (int64_t i = 0; i < splits_.size(); ++i) {
        const auto binFeature = splits_[i];
        const auto border = grid_->condition(binFeature.featureId_, binFeature.conditionId_);
        const auto val = xRef[grid_->origFeatureIndex(binFeature.featureId_)];
        probs[i] = 1./(1. + exp(-(val - border)));//*1000));
    }
    double res = 0;
    for (uint32_t b = 0; b < leaves_.dim(); ++b) {
        double value = bitVec[b];
        for (int f = 0; f < probs.size(); ++f) {
            if (((b >> f) & 1) != 0) {
                value *= probs[f];
            }
        }
        res += value;
    }
    return res;
}

void ObliviousTree::grad(const Vec& x, Vec to) {
    std::vector<uint32_t> masks(x.dim());
    for (int i = 0; i < splits_.size(); ++i) {
        const auto binFeature = splits_[i];
        masks[grid_->origFeatureIndex(binFeature.featureId_)] += (1 << (i));
    }
    std::vector<double> probs(splits_.size());

    for (int64_t i = 0; i < probs.size(); i++) {
        const auto binFeature = splits_[i];
        const auto border = grid_->condition(binFeature.featureId_, binFeature.conditionId_);
        const auto val = x.get(grid_->origFeatureIndex(binFeature.featureId_));
        probs[i] = 1./(1. + exp(-(val - border)));//*1000));
    }

    std::vector<double> probs_mult_buff(leaves_.dim(), 1);

    for (uint32_t b = 0; b < leaves_.dim(); ++b) {
        for (int f = 0; f < probs.size(); f++) {
            if (b >> f == 0) break;
            if (((b >> f) & 1) != 0) {
                probs_mult_buff[b] *= probs[f];
            }
        }
    }

    auto toRef = to.arrayRef();

    for (int i = 0; i < to.dim(); ++i) {

        double res = 0;

        for (uint32_t b = 0; b < leaves_.dim(); ++b) {
            uint32_t mask_b = masks[i] & b;
            if (mask_b == 0) {
                continue;
            }

            double value = bitVec[b] * probs_mult_buff[b];
            double diffCf = 0;

            for (int f = 0; f < probs.size(); f++) {
                if (mask_b >> f == 0) break;
                if (((mask_b >> f) & 1) != 0) {
                    diffCf += 1 - probs[f];
                }
            }
            res += value * diffCf;//*1000;
        }
        toRef[i] += res;
    }
}

uint32_t ObliviousTree::bits(uint32_t i) {
    uint32_t result = 0;
    while (i != 0) {
        result += (i & 1);
        i >>= 1;
    }
    return result;
}
