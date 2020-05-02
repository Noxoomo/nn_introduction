#pragma once

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

    Monom(PolynomStructure structure, std::vector<double> values)
            : Structure_(std::move(structure))
            , Values_(std::move(values)) {

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
    static MonomPtr createMonom(MonomType monomType, PolynomStructure structure, std::vector<double> values);

    virtual ~Monom() = default;

public:
    PolynomStructure Structure_;
    std::vector<double> Values_;
};

class SigmoidProbMonom : public Monom {
public:
    SigmoidProbMonom() = default;

    SigmoidProbMonom(PolynomStructure structure, std::vector<double> values)
            : Monom(std::move(structure), std::move(values)) {

    }

    Monom::MonomType getMonomType() const override;

    void Forward(double lambda, ConstVecRef<float> features, VecRef<float> dst) const override;
    void Backward(double lambda, ConstVecRef<float> features, ConstVecRef<float> outputsDer, VecRef<float> featuresDer) const override;
};
