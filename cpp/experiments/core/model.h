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
    }

    virtual torch::Tensor  forward(torch::Tensor x) = 0;

    torch::Device device() const {
        return this->parameters().data()->device();
    }

    // WTF torch, this should be default behaviour
    void train(bool on = true) override {
        for (auto &param : parameters()) {
            param.set_requires_grad(on);
        }
        torch::nn::Module::train(on);
    }

private:
    torch::Tensor dummy_;
};

using ModelPtr = std::shared_ptr<Model>;


class ScaleModel : public Model {
public:
    ScaleModel(torch::Device device);

    torch::Tensor forward(torch::Tensor x) override ;
    void printScale();
    ~ScaleModel() override = default;

private:
    torch::Tensor classifierScale_;
};

// Classifier

class Classifier : public Model {
public:

    explicit Classifier(ModelPtr classifier) {
        classifier_ = register_module("classifier_", std::move(classifier));
        scaleModel_.reset(new ScaleModel(classifier_->device()));
        scaleModel_ = register_module("scale_", scaleModel_);
//        classifierScale_ = torch::ones({1}, opts).to(baseline_->device());
    }

    explicit Classifier(ModelPtr classifier, ModelPtr baseline) {
        classifier_ = register_module("classifier_", std::move(classifier));
        baseline_ = register_module("baseline_", std::move(baseline));
        scaleModel_.reset(new ScaleModel(classifier_->device()));
        scaleModel_ = register_module("scale_", scaleModel_);
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

    virtual void enableScaleTrain(bool flag) {
       scaleModel_->train(flag);
    }

    void printScale() {
        dynamic_cast<ScaleModel&>(*scaleModel_).printScale();
    }

    torch::Tensor forward(torch::Tensor x) override;

private:
    ModelPtr classifier_;
    ModelPtr baseline_;
    ModelPtr scaleModel_;

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

// ConvModel

class ConvModel : public Model {
public:
    ConvModel(ModelPtr conv,
              ClassifierPtr classifier) {
        conv_ = register_module("conv_", std::move(conv));
        classifier_ = register_module("classifier_", std::move(classifier));
    }

    virtual ModelPtr conv() {
        return conv_;
    };

    virtual ClassifierPtr classifier() {
        return classifier_;
    };

    torch::Tensor forward(torch::Tensor x) override;

    void train(bool on = true) override;

private:
    ModelPtr conv_;
    ClassifierPtr classifier_;
};

using ConvModelPtr = std::shared_ptr<ConvModel>;

class MLP : public Model {
public:
    MLP(const std::vector<int>& sizes);

    torch::Tensor forward(torch::Tensor x) override;

    ~MLP() override = default;

private:
    std::vector<torch::nn::Linear> layers_;
};

class LinearCifarClassifier : public Model {
public:
    LinearCifarClassifier(int dim);

    torch::Tensor forward(torch::Tensor x) override;

    ~LinearCifarClassifier() override = default;

private:
    torch::nn::Linear fc1_{nullptr};
};

class SigmoidLinearCifarClassifier : public Model {
public:
    SigmoidLinearCifarClassifier(int dim);

    torch::Tensor forward(torch::Tensor x) override;

    ~SigmoidLinearCifarClassifier() override = default;

private:
    torch::nn::Linear fc1_{nullptr};
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

inline ModelPtr makeCifarLinearClassifier(int inputDim) {
    return std::make_shared<LinearCifarClassifier>(inputDim);
}

inline ModelPtr makeCifarBias() {
    return std::make_shared<Bias>(10);
}

template <class Impl, class... Args>
inline ClassifierPtr makeClassifier(Args&&... args) {
    return std::make_shared<Classifier>(std::make_shared<Impl>(std::forward<Args>(args)...));
}

template <class Impl, class... Args>
inline ClassifierPtr makeClassifierWithBaseline(ModelPtr baseline, Args&&... args) {
    return std::make_shared<Classifier>(std::make_shared<Impl>(std::forward<Args>(args)...), baseline);
}

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params);
ClassifierPtr createClassifier(int numClasses, const json& params);

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

}
