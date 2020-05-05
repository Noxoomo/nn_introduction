#pragma once

#include "tensor_pair_dataset.h"
#include <util/json.h>

#include <torch/torch.h>

#include <memory>
#include <utility>
#include <unordered_map>

namespace experiments {

// Model

class Model : public torch::nn::Module {
public:
    Model() {
        // only need this parameter in order to correctly track
        // device of those models which don't have real parameters
        dummy_ = torch::zeros({});
        dummy_ = register_parameter("dummy", dummy_, false);
//        lastNonlinearity_ = true;
//        lastBias_ = torch::zeros({});
//        lastBias_ = register_parameter("lastBias", lastBias_, false);
    }

    virtual torch::Tensor forward(torch::Tensor x) = 0;

//    virtual void lastNonlinearity(bool on) {
//        lastNonlinearity_ = on;
//    }

    void setLastBias(torch::Tensor b) {
//        lastBias_ = b.to(dummy_.device());
    }

    torch::Device device() const {
        return this->parameters().data()->device();
    }

    // WTF torch, this should be default behaviour
    void train(bool on) override {
        for (auto &param : parameters()) {
            param.set_requires_grad(on);
        }
        torch::nn::Module::train(on);
    }

protected:
//    bool lastNonlinearity_;
//    torch::Tensor lastBias_;

private:
    torch::Tensor dummy_;
};

using ModelPtr = std::shared_ptr<Model>;

// Classifier

class Classifier : public Model {
public:

    explicit Classifier(ModelPtr classifier) {
        classifier_ = register_module("classifier_", std::move(classifier));
//        torch::TensorOptions opts;
//        opts = opts.dtype(torch::kFloat32);
//        opts = opts.requires_grad(true);
//        opts = opts.device(classifier_->device());
//        classifierScale_ = register_parameter("scale_", torch::ones({1}, opts));
    }

    explicit Classifier(ModelPtr classifier, ModelPtr baseline) {
        classifier_ = register_module("classifier_", std::move(classifier));
        baseline_ = register_module("baseline_", std::move(baseline));
//        torch::TensorOptions opts;
//        opts = opts.dtype(torch::kFloat32);
//        opts = opts.requires_grad(true);
//        opts = opts.device(classifier_->device());
//        classifierScale_ = register_parameter("scale_", torch::ones({1}, opts));
    }

    virtual ModelPtr classifier() {
        return classifier_;
    }

    virtual ModelPtr baseline() {
        return baseline_;
    }

    virtual void enableBaselineTrain(bool flag) {
        if (baseline_) {
            baseline_->train(flag);
        }
    }

    virtual void setScale(double scale) {
//        classifierScale_[0] = scale;
    }

    virtual void enableScaleTrain(bool flag) {
//        classifierScale_.set_requires_grad(flag);
    }


    void printScale() {
//        std::cout << "classifier scale = " << classifierScale_[0]
//                  << ", requires_grad = " << classifierScale_.requires_grad()
//                  << ", grad = " << classifierScale_.grad()
//                  << ", device = " << classifierScale_.device() << std::endl;
    }

    torch::Tensor forward(torch::Tensor x) override;

private:
    ModelPtr classifier_;
    ModelPtr baseline_;
//    torch::Tensor classifierScale_;

};

using ClassifierPtr = std::shared_ptr<Classifier>;

// ZeroClassifier

class ZeroClassifier : public Model {
public:
    ZeroClassifier(int numClasses);

    torch::Tensor forward(torch::Tensor x) override;

    ~ZeroClassifier() override = default;

private:
    int numClasses_;
};

class IdClassifier : public Model {
public:
    IdClassifier() = default;

    torch::Tensor forward(torch::Tensor x) override {
//        std::cout << "forwarding Id" << std::endl;
        return x;
    }

    ~IdClassifier() override = default;
};

// ConvModel

class ConvModel : public Model {
public:
    ConvModel(ModelPtr conv,
              ClassifierPtr classifier) {
        conv_ = register_module("conv_", std::move(conv));
        classifier_ = register_module("classifier_", std::move(classifier));
    }

    ModelPtr conv() {
        return conv_;
    };

    ClassifierPtr classifier() {
        return classifier_;
    }

    torch::Tensor forward(torch::Tensor x) override {
//        std::cout << "forwarding conv" << std::endl;
        if (conv_) {
//            std::cout << "forwarding conv_" << std::endl;
            x = conv_->forward(x);
        }
        if (classifier_) {
//            std::cout << "forwarding classifier_" << std::endl;
            x = classifier_->forward(x);
        }
        return x;
    }

    void train(bool on) override {
        if (conv_) {
            conv_->train(on);
        }
        if (classifier_) {
            classifier_->train(on);
        }
    }

private:
    ModelPtr conv_;
    ClassifierPtr classifier_;
};

// EmModel

class EmModel : public Model {
public:
    EmModel(ModelPtr eStepModel,
            ModelPtr mStepModel) {
        eStepModel_ = register_module("eStepModel_", std::move(eStepModel));
        mStepModel_ = register_module("mStepModel_", std::move(mStepModel));
    }

    ModelPtr eStepModel() {
        return eStepModel_;
    };

    ModelPtr mStepModel() {
        return mStepModel_;
    };

    torch::Tensor forward(torch::Tensor x) override;

    void train(bool on) override;

private:
    ModelPtr eStepModel_;
    ModelPtr mStepModel_;
};

using EmModelPtr = std::shared_ptr<EmModel>;

class MLP : public Model {
public:
    MLP(const std::vector<int>& sizes);

    torch::Tensor forward(torch::Tensor x) override;

    ~MLP() override = default;

private:
    std::vector<torch::nn::Linear> layers_;
};

class Bias : public Model {
public:
    Bias(int dim);

    torch::Tensor forward(torch::Tensor x) override;

    ~Bias() override = default;

private:
    torch::Tensor bias_;
};

// Utils

inline ModelPtr makeCifarBias() {
    return std::make_shared<Bias>(10);
}

template<class Impl, class... Args>
inline ClassifierPtr makeClassifier(Args&&... args) {
    return std::make_shared<Classifier>(std::make_shared<Impl>(std::forward<Args>(args)...));
}

template<class Impl, class... Args>
inline ClassifierPtr makeClassifierWithBaseline(ModelPtr baseline, Args&&... args) {
    return std::make_shared<Classifier>(std::make_shared<Impl>(std::forward<Args>(args)...), baseline);
}

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params);

ClassifierPtr createClassifier(int numClasses, const json& params);

ModelPtr createModel(const json& params);

inline torch::Tensor correctDevice(torch::Tensor x, const torch::Device& to) {
    if (x.device() != to) {
        return x.to(to);
    } else {
        return std::move(x);
    }
}

inline torch::Tensor correctDevice(torch::Tensor x, const ModelPtr& to) {
    return correctDevice(std::move(x), to->device());
}

inline torch::Tensor correctDevice(torch::Tensor x, const Model& to) {
    return correctDevice(std::move(x), to.device());
}

torch::Device getDevice(const std::string &deviceType);

}