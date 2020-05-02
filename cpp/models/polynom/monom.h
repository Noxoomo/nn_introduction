#pragma once

#include <vector>
#include <memory>

#include <core/vec.h>
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

struct PolynomStructure {
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

// taken from https://stackoverflow.com/a/38140932/2304212
inline void hash_combine(std::size_t& seed) { }

template <typename T, typename... Rest>
inline void hash_combine(std::size_t& seed, const T& v, Rest... rest) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    hash_combine(seed, rest...);
}

template <>
struct std::hash<std::tuple<PolynomStructure, int>> {
    inline size_t operator()(const std::tuple<PolynomStructure, int>& tuple) const {
        std::size_t ret = 0;
        hash_combine(ret, std::get<0>(tuple), std::get<1>(tuple));
        return ret;
    }
};

// TBD: Not sure if we should make it unique or shared ptr
class Monom;
using MonomPtr = std::shared_ptr<Monom>;

class Monom {
public:
    enum class MonomType {
        SigmoidProbMonom,
        ExpProbMonom,
        LinearMonom,
    };

    friend struct Polynom;

public:
    Monom() = default;

    Monom(PolynomStructure structure, std::vector<double> values, int origFId = -1)
            : Structure_(std::move(structure))
            , Values_(std::move(values))
            , origFId_(origFId) {

    }

    int OutDim() const {
        return (int)Values_.size();
    }

    //forward/backward will append to dst
    virtual void Forward(double lambda, ConstVecRef<float> features, VecRef<float> dst) const = 0;
    virtual void Backward(double lambda, ConstVecRef<float> features, ConstVecRef<float> outputsDer, VecRef<float> featuresDer) const = 0;

    virtual MonomType getMonomType() const = 0;

    static MonomType getMonomType(const std::string& strMonomType);

    static MonomPtr createMonom(MonomType monomType);

    static MonomPtr createMonom(MonomType monomType, PolynomStructure structure, std::vector<double> values, int origFId);

    virtual ~Monom() = default;

public:
    PolynomStructure Structure_;
    std::vector<double> Values_;
    int origFId_;
};
