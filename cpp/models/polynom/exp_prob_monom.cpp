#include "exp_prob_monom.h"

Monom::MonomType ExpProbMonom::getMonomType() const {
    return Monom::MonomType::ExpProbMonom;
}

void ExpProbMonom::Forward(double lambda, ConstVecRef<float> features, VecRef<float> dst) const {
    double trueLogProb = 0;
    bool zeroProb = false;

    for (const auto& split : Structure_.Splits) {
        const double val = -lambda * features[split.Feature];
        const double expVal = 1.0f - exp(val);
        if (std::isfinite(log(expVal))) {
            trueLogProb += log(expVal);
        } else {
            zeroProb = true;
            break;
        }
    }

    double p = 0.0f;
    if (!zeroProb) {
        p = exp(trueLogProb);
    }
    for (int dim = 0; dim < (int)dst.size(); ++dim) {
        dst[dim] += p * Values_[dim];
    }
}

void ExpProbMonom::Backward(double lambda, ConstVecRef<float> features, ConstVecRef<float> outputsDer,
                            VecRef<float> featuresDer) const {
    std::vector<double> logProbs;
    logProbs.resize(Structure_.Splits.size(), 0.f);
    std::vector<double> vals;
    vals.resize(Structure_.Splits.size(), 0.f);

    double derMultiplier = 0;
    for (size_t dim = 0; dim < Values_.size(); ++dim) {
        derMultiplier += 1e5 * Values_[dim] * outputsDer[dim];
    }

    double totalLogProb = 0;
    bool zeroProb = false;

    for (int i = 0; i < (int)Structure_.Splits.size(); ++i) {
        const auto& split = Structure_.Splits[i];
        vals[i] = -lambda * features[split.Feature];
        const double expVal = 1.0f - exp(vals[i]);
        if (std::isfinite(log(expVal))) {
            logProbs[i] += log(expVal);
        } else {
            zeroProb = true;
            break;
        }
        totalLogProb += logProbs[i];
    }

    for (int i = 0; i < (int)Structure_.Splits.size(); ++i) {
        const auto& split = Structure_.Splits[i];
        if (!zeroProb) {
            const double monomDer = exp(totalLogProb - logProbs[i] + log(lambda) + vals[i]);
            featuresDer[split.Feature] += monomDer * derMultiplier;
        }
    }
}
