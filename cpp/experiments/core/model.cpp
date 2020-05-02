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

torch::Tensor experiments::EmModel::forward(torch::Tensor x) {
    if (eStepModel_) {
//        std::cout << "forwarding e model" << std::endl;
        x = eStepModel_->forward(x);
    }
    if (mStepModel_) {
//        std::cout << "forwarding m model" << std::endl;
        x = mStepModel_->forward(x);
    }
    return x;
}

void experiments::EmModel::train(bool on) {
    if (eStepModel_) {
        eStepModel_->train(on);
    }
    if (mStepModel_) {
        mStepModel_->train(on);
    }
}

torch::Tensor experiments::Classifier::forward(torch::Tensor x) {
    x = x.view({x.size(0), -1});
    auto result = classifier_->forward(x);
//    result *= classifierScale_;
//    if (baseline_) {
//        result = correctDevice(result, baseline_->device());
////        x = torch::sigmoid(x);
//        result += baseline_->forward(x);
//    }
//    result *= classifierScale_;
    return result;
}

ZeroClassifier::ZeroClassifier(int numClasses)
        : numClasses_(numClasses) {

}

torch::Tensor ZeroClassifier::forward(torch::Tensor x) {
//    std::cout << "forwarding zero" << std::endl;
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

// TODO move it somewhere
torch::Device getDevice(const std::string& device) {
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

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params) {
    auto model = _createConvLayers(inputShape, params);
    auto device = getDevice(params[DeviceKey]);
    model->to(device);
    return model;
}

static ModelPtr _createClassifier(int numClasses, const json& params) {
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
        return std::make_shared<PolynomModel>(std::move(polynom));
    } else if (archType == "Id") {
        return std::make_shared<IdClassifier>();
    }

    std::string errMsg("Unsupported baseline classifier type");
    throw std::runtime_error(errMsg + " " + archType);
}

ClassifierPtr createClassifier(const json& params) {
    ModelPtr mainClassifier{nullptr};
    ModelPtr baselineClassifier{nullptr};

    int numClasses = params["num_classes"];

    if (params.count(ClassifierMainKey)) {
        mainClassifier = _createClassifier(numClasses, params[ClassifierMainKey]);
    } else {
        mainClassifier = std::make_shared<ZeroClassifier>(numClasses);
    }
    auto device = getDevice(params[ClassifierMainKey][DeviceKey]);
    mainClassifier->to(device);

    if (params.count(ClassifierBaselineKey)) {
        baselineClassifier = _createClassifier(numClasses, params[ClassifierBaselineKey]);
        auto device = getDevice(params[ClassifierBaselineKey][DeviceKey]);
        baselineClassifier->to(device);
    }

    if (baselineClassifier.get() != nullptr) {
        return std::make_shared<Classifier>(mainClassifier, baselineClassifier);
    } else {
        return std::make_shared<Classifier>(mainClassifier);
    }
}

ModelPtr createConvModel(const json& convModelParams) {
    ModelPtr conv;
    ClassifierPtr classifier;

    if (convModelParams.contains("conv")) {
        const json& convParams = convModelParams["conv"];
        conv = createConvLayers({}, convParams);
    }

    if (convModelParams.contains("classifier")) {
        const json& classifierParams = convModelParams["classifier"];
        classifier = createClassifier(classifierParams);
    }

    return std::make_shared<ConvModel>(std::move(conv), std::move(classifier));
}

ModelPtr createEmModel(const json& emModelParams) {
    ModelPtr eModel = createModel(emModelParams["e_model"]);
    auto device = getDevice(emModelParams["e_model"][DeviceKey]);
    eModel->to(device);

    ModelPtr mModel = createModel(emModelParams["m_model"]);
    device = getDevice(emModelParams["m_model"][DeviceKey]);
    mModel->to(device);

    return std::make_shared<EmModel>(std::move(eModel), std::move(mModel));
}

ModelPtr createModel(const json& modelParams) {
    ModelPtr model;

    const std::string& arch = modelParams["model_arch"];
    if (arch == "em_model") {
        model = createEmModel(modelParams);
    } else if (arch == "conv_model") {
        model = createConvModel(modelParams);
    } else if (arch == "Id") {
        model = std::make_shared<IdClassifier>();
    } else {
        throw std::runtime_error("Unknown model arch " + arch);
    }

    if (modelParams["load_from_checkpoint"] == true) {
        const std::string& path = modelParams["checkpoint_file"];
        std::cout << "Loading model from '" << path << "'" << std::endl;
        torch::load(model, path);
    }

    return model;
}

}
