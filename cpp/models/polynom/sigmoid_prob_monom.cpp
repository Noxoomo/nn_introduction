#include "sigmoid_prob_monom.h"

Monom::MonomType SigmoidProbMonom::getMonomType() const {
    return Monom::MonomType::SigmoidProbMonom;
}

void SigmoidProbMonom::Forward(double lambda, ConstVecRef<float> features, VecRef<float> dst) const {
    double trueLogProb = 0;
    for (const auto& split : Structure_.Splits) {
        const double val = -lambda * (features[split.Feature] - split.Condition);
//    log(1.0 / (1.0 + exp(-val))) = -log(1.0 + exp(-val));

        const double expVal = exp(val);
        if (std::isfinite(expVal)) {
            trueLogProb -= log(1.0 + expVal);
        } else {
            trueLogProb -= val;
        }
    }
    const double p = exp(trueLogProb);
    for (int dim = 0; dim < (int)dst.size(); ++dim) {
        dst[dim] += p * Values_[dim];
    }
}

void SigmoidProbMonom::Backward(double lambda,
                                ConstVecRef<float> features,
                                ConstVecRef<float> outputsDer,
                                VecRef<float> featuresDer) const {
    std::vector<double> logProbs;
    logProbs.resize(Structure_.Splits.size(), 0.0f);

    double totalLogProb = 0;

    for (int i = 0; i < (int)Structure_.Splits.size(); ++i) {
        const auto& split = Structure_.Splits[i];
        const double val = -lambda * (features[split.Feature] - split.Condition);
//    log(1.0 / (1.0 + exp(-val))) = -log(1.0 + exp(-val));

        const double expVal = exp(val);
        if (std::isfinite(expVal)) {
            logProbs[i] -= log(1.0 + expVal);
        } else {
            logProbs[i] -= val;
        }
        totalLogProb += logProbs[i];
    }
    const double p = exp(totalLogProb);

    double tmp = 0;
    for (size_t dim = 0; dim < Values_.size(); ++dim) {
        tmp += Values_[dim] * outputsDer[dim];
    }

    for (int i = 0; i < (int)Structure_.Splits.size(); ++i) {
        const auto& split = Structure_.Splits[i];
        const double featureProb = exp(logProbs[i]);
        const double monomDer = p * (1.0 - featureProb);
        featuresDer[split.Feature] += monomDer * tmp;
    }
}
