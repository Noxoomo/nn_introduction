#include "model.h"
using namespace experiments;

LinearCifarClassifier::LinearCifarClassifier(int dim) {
    fc1_ = register_module("fc1_", torch::nn::Linear(dim, 10));
}

torch::Tensor LinearCifarClassifier::forward(torch::Tensor x) {
    return fc1_->forward(x.view({x.size(0), -1}));
}

SigmoidLinearCifarClassifier::SigmoidLinearCifarClassifier(int dim) {
    fc1_ = register_module("fc1_", torch::nn::Linear(dim, 10));
}

torch::Tensor SigmoidLinearCifarClassifier::forward(torch::Tensor x) {
    return fc1_->forward(torch::sigmoid(x.view({x.size(0), -1})));
}

torch::Tensor experiments::ConvModel::forward(torch::Tensor x) {
    x = conv()->forward(x);
    return classifier()->forward(x);
}

void experiments::ConvModel::train(bool on) {
    conv()->train(on);
    classifier()->train(on);
}

torch::Tensor experiments::Classifier::forward(torch::Tensor x) {
    x = x.view({x.size(0), -1});
    auto result = classifier_->forward(x);
    if (baseline_)  {
        result *= classifierScale_;
        result += baseline_->forward(x);
    }
    return result;
}

Bias::Bias(int dim) {
    bias_ = register_parameter("bias_", torch::zeros({dim}, torch::kFloat32));

}
torch::Tensor Bias::forward(torch::Tensor x) {
    torch::TensorOptions options;
    options = options.device(x.device());
    options = options.dtype(torch::kFloat32);
    torch::Tensor result = torch::zeros({x.size(0), bias_.size(0)}, options);
    result += bias_;
    return result;
}
