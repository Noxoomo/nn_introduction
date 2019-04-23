#pragma once

#include <catboost_wrapper.h>
#include <unordered_map>
#include <util/array_ref.h>
#include <util/city.h>

struct BinarySplit {
    int Feature = 0;
    float Condition = 0;
    bool operator==(const BinarySplit& rhs) const {
        return std::tie(Feature, Condition) == std::tie(rhs.Feature, rhs.Condition);
    }
    bool operator!=(const BinarySplit& rhs) const {
        return !(rhs == *this);
    }

    bool operator<(const BinarySplit& rhs) const {
        return std::tie(Feature, Condition) < std::tie(rhs.Feature, rhs.Condition);
    }
    bool operator>(const BinarySplit& rhs) const {
        return rhs < *this;
    }
    bool operator<=(const BinarySplit& rhs) const {
        return !(rhs < *this);
    }
    bool operator>=(const BinarySplit& rhs) const {
        return !(*this < rhs);
    }
};


struct  PolynomStructure {
    std::vector<BinarySplit> Splits;


    uint32_t GetDepth() const {
        return Splits.size();
    }

    void AddSplit(const BinarySplit& split) {
        Splits.push_back(split);
    }

    bool operator==(const PolynomStructure& rhs) const {
        return std::tie(Splits) == std::tie(rhs.Splits);
    }

    bool operator!=(const PolynomStructure& rhs) const {
        return !(rhs == *this);
    }

    uint64_t GetHash() const {
        return VecCityHash(Splits);
    }

    bool IsSorted() const {
        for (uint32_t i = 1; i < Splits.size(); ++i) {
            if (Splits[i] <= Splits[i - 1]) {
                return false;
            }
        }
        return true;
    }

    bool HasDuplicates() const {
        for (uint32_t i = 1; i < Splits.size(); ++i) {
            if (Splits[i] == Splits[i - 1]) {
                return true;
            }
        }
        return false;
    }

};

template <>
struct std::hash<PolynomStructure> {
    inline size_t operator()(const PolynomStructure& value) const {
        return value.GetHash();
    }
};




struct TStat {
    std::vector<double> Value;
    double Weight = -1;
};


// sum v * Prod [x _i > c_i]
class PolynomBuilder {
public:

    void AddTree(const TSymmetricTree& tree);

    PolynomBuilder& AddEnsemble(const TEnsemble& ensemble) {
        for (const auto& tree : ensemble.Trees) {
            AddTree(tree);
        }
        return *this;
    }

    std::unordered_map<PolynomStructure, TStat> Build() {
        return EnsemblePolynoms;
    }
private:
    std::unordered_map<PolynomStructure, TStat> EnsemblePolynoms;
};


struct Monom {
    PolynomStructure Structure_;
    std::vector<double> Values_;

    int OutDim() const {
        return Values_.size();
    }

    Monom(PolynomStructure structure, std::vector<double> values)
    : Structure_(std::move(structure))
    , Values_(std::move(values)) {

    }

    Monom() {

    }

    //forward/backward will append to dst
    void Forward(double lambda, ConstVecRef<float> features, VecRef<float> dst) const;
    void Backward(double lambda, ConstVecRef<float> features, ConstVecRef<float> outputsDer, VecRef<float> featuresDer) const;
};

struct Polynom {
    std::vector<Monom> Ensemble_;
    double Lambda_  = 1.0;


    Polynom(const std::unordered_map<PolynomStructure, TStat>& polynom) {
        for (const auto& [structure, stat] : polynom) {
            Ensemble_.emplace_back(structure, stat.Value);
        }
    }

    Polynom() {

    }
    //forward/backward will append to dst
    void Forward(ConstVecRef<float> features, VecRef<float> dst) const;
    void Backward(ConstVecRef<float> features, ConstVecRef<float> outputsDer, VecRef<float> featuresDer) const;

    int OutDim() const {
        return Ensemble_.empty() ? 0 : Ensemble_.back().OutDim();
    }


};
