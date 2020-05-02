#include "monom.h"

#include "sigmoid_prob_monom.h"
#include "exp_prob_monom.h"
#include "linear_monom.h"

template<class T, class... Args>
inline static MonomPtr _makeMonomPtr(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

Monom::MonomType Monom::getMonomType(const std::string &strMonomType) {
    if (strMonomType == "SigmoidProbMonom") {
        return MonomType::SigmoidProbMonom;
    } else if (strMonomType == "ExpProbMonom") {
        return MonomType::ExpProbMonom;
    } else if (strMonomType == "LinearMonom") {
        return MonomType::LinearMonom;
    } else {
            throw std::runtime_error("Unsupported monom type '" + strMonomType + "'");
    }
}

MonomPtr Monom::createMonom(Monom::MonomType monomType) {
    if (monomType == Monom::MonomType::SigmoidProbMonom) {
        return _makeMonomPtr<SigmoidProbMonom>();
    } else if (monomType == Monom::MonomType::ExpProbMonom) {
        return _makeMonomPtr<ExpProbMonom>();
    } else if (monomType == Monom::MonomType::LinearMonom) {
        return _makeMonomPtr<LinearMonom>();
    } else {
        throw std::runtime_error("Unsupported monom type");
    }
}

MonomPtr Monom::createMonom(Monom::MonomType monomType, PolynomStructure structure,
                            std::vector<double> values, int origFId) {
    if (monomType == Monom::MonomType::SigmoidProbMonom) {
        return _makeMonomPtr<SigmoidProbMonom>(std::move(structure), std::move(values));
    } else if (monomType == Monom::MonomType::ExpProbMonom) {
        return _makeMonomPtr<ExpProbMonom>(std::move(structure), std::move(values));
    } else if (monomType == Monom::MonomType::LinearMonom) {
        return _makeMonomPtr<LinearMonom>(std::move(structure), std::move(values), origFId);
    } else {
        throw std::runtime_error("Unsupported monom type");
    }
}