#include "model.h"

#include "networks/vgg.h"
#include "networks/lenet.h"
#include "networks/resnet.h"
#include "networks/mobile_net_v2.h"
#include "networks/small_net.h"
#include "polynom_model.h"
#include "params.h"

#include <models/polynom/polynom.h>

#include <algorithm>
#include <stdexcept>

namespace experiments {

MLP::MLP(const std::vector<int>& sizes) {
    for (int i = 0; i < sizes.size() - 1; i++) {
        int in = sizes[i];
        int out = sizes[i + 1];
        auto layer = register_module("layer_" + std::to_string(i),
                torch::nn::Linear(in, out));
        layers_.emplace_back(std::move(layer));
    }
}

torch::Tensor MLP::forward(torch::Tensor x) {
    x = correctDevice(x, *this);
    for (int i = 0; i < (int)layers_.size(); ++i) {
        x = layers_[i]->forward(x);
        if (i != (int)layers_.size() - 1) {
            x = torch::relu(x);
        }
    }
    return x;
}

LinearCifarClassifier::LinearCifarClassifier(int dim) {
    fc1_ = register_module("fc1_", torch::nn::Linear(dim, 10));
}

torch::Tensor LinearCifarClassifier::forward(torch::Tensor x) {
    x = correctDevice(x, *this);
    return fc1_->forward(x.view({x.size(0), -1}));
}

SigmoidLinearCifarClassifier::SigmoidLinearCifarClassifier(int dim) {
    fc1_ = register_module("fc1_", torch::nn::Linear(dim, 10));
}

torch::Tensor SigmoidLinearCifarClassifier::forward(torch::Tensor x) {
    x = correctDevice(x, *this);
    return fc1_->forward(torch::sigmoid(x.view({x.size(0), -1})));
}

torch::Tensor experiments::ConvModel::forward(torch::Tensor x) {
    x = conv()->forward(x);
    x = classifier()->forward(x);
    return x;
}

void experiments::ConvModel::train(bool on) {
    conv()->train(on);
    classifier()->train(on);
}

torch::Tensor experiments::Classifier::forward(torch::Tensor x) {
    x = x.view({x.size(0), -1});
    auto result = classifier_->forward(x);
    result = scaleModel_->forward(result);
    if (baseline_) {
        result = correctDevice(result, baseline_->device());
        result += baseline_->forward(x);
    }
    return result;
}

ZeroClassifier::ZeroClassifier(int numClasses)
        : numClasses_(numClasses) {

}

torch::Tensor ZeroClassifier::forward(torch::Tensor x) {
    return torch::zeros({x.size(0), numClasses_}, torch::kFloat32).to(this->parameters().data()->device());
}

Bias::Bias(int dim) {
    bias_ = register_parameter("bias_", torch::zeros({dim}, torch::kFloat32));
}

torch::Tensor Bias::forward(torch::Tensor x) {
    torch::TensorOptions options;
    options = options.device(bias_.device());
    options = options.dtype(torch::kFloat32);
    torch::Tensor result = torch::zeros({x.size(0), bias_.size(0)}, options);
    result += bias_;
    return result;
}

// Utils

static ModelPtr _createConvLayers(const std::vector<int>& inputShape, const json& params) {
    std::string modelName = params[ModelArchKey];

    if (modelName == "LeNet") {
        return lenet::createConvLayers(inputShape, params);
    } else if (modelName == "VGG") {
        return vgg::createConvLayers(inputShape, params);
    } else if (modelName == "ResNet") {
        return resnet::createConvLayers(inputShape, params);
    } else if (modelName == "MobileNetV2") {
        return mobile_net_v2::createConvLayers(inputShape, params);
    } else if (modelName == "SmallNet") {
        return small_net::createConvLayers(inputShape, params);
    }

    std::string errMsg("Unsupported model architecture");
    throw std::runtime_error(errMsg + " " + modelName);
}

// TODO move it somewhere
static torch::Device getDevice(const std::string& device) {
    auto ls = splitByDelim(device, ':');
    if (ls[0] == "GPU") {
        if (ls.size() > 1) {
            return torch::Device(torch::kCUDA, std::stoi(ls[1]));
        } else {
            return torch::kCUDA;
        }
    } else {
        return torch::kCPU;
    }
}

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params) {
    auto model = _createConvLayers(inputShape, params);
    auto device = getDevice(params[DeviceKey]);
    model->to(device);
    return model;
}

static ModelPtr _createInnerModel(int numClasses, const json& params) {
    std::string archType = params[ModelArchKey];

    if (archType == "MLP") {
        std::vector<int> sizes = params[DimsKey];
        return std::make_shared<MLP>(sizes);
    } else if (archType == "Polynom") {
        PolynomPtr polynom = std::make_shared<Polynom>();
        polynom->Lambda_ = params[LambdaKey];
        {
            auto monomType = Monom::getMonomType(params[MonomTypeKey]);
            auto emptyMonom = Monom::createMonom(monomType);
            emptyMonom->Structure_.Splits.push_back({0, 0});
            emptyMonom->Values_.resize(numClasses);
            polynom->Ensemble_.push_back(std::move(emptyMonom));
        }
        const std::shared_ptr<PolynomModel> &result = std::make_shared<PolynomModel>(std::move(polynom));
        return result;
    }

    std::string errMsg("Unsupported baseline classifier type");
    throw std::runtime_error(errMsg + " " + archType);
}

ClassifierPtr createClassifier(int numClasses, const json& params) {
    ModelPtr mainClassifier{nullptr};
    ModelPtr baselineClassifier{nullptr};

    if (params.count(ClassifierMainKey)) {
        mainClassifier = _createInnerModel(numClasses, params[ClassifierMainKey]);
        auto device = getDevice(params[ClassifierMainKey][DeviceKey]);
        mainClassifier->to(device);
    } else {
        mainClassifier = std::make_shared<ZeroClassifier>(numClasses);
    }

    if (params.count(ClassifierBaselineKey)) {
        baselineClassifier = _createInnerModel(numClasses, params[ClassifierBaselineKey]);
        auto device = getDevice(params[ClassifierBaselineKey][DeviceKey]);
        baselineClassifier->to(device);
    }

    if (baselineClassifier.get() != nullptr) {
        return std::make_shared<Classifier>(mainClassifier, baselineClassifier);
    } else {
        return std::make_shared<Classifier>(mainClassifier);
    }
}

    void ScaleModel::printScale()  {
        std::cout << "classifier scale = " << classifierScale_.norm(1) << std::endl;
        std::cout << "classifier scale requires grad = " << classifierScale_.requires_grad() << std::endl;
    }

    torch::Tensor ScaleModel::forward(torch::Tensor x) {
        const int batchSize = x.size(0);
        auto tmp = torch::zeros({batchSize}, torch::kFloat32).to(x.device());
        auto tmp2 = torch::ones({batchSize}, torch::kFloat32).to(x.device());

        auto res = torch::cat({tmp2 * classifierScale_, tmp}, 0).reshape({2, batchSize}).transpose(0, 1).contiguous();
//        std::cout << res.clone() << std::endl;
        return res;
//        return  x * classifierScale_;
    }

    ScaleModel::ScaleModel(torch::Device device)  {
        torch::TensorOptions opts;
        opts = opts.dtype(torch::kFloat32);
        opts = opts.requires_grad(true);
//        classifierScale_ = torch::ones({1}, opts).to(baseline_->device());
        classifierScale_ = torch::ones({1}, opts).to(device) * 100;
        classifierScale_ = register_parameter("scale_", classifierScale_);
        VERIFY(classifierScale_.is_variable(), "wtf. here should be autograd");

    }
}
