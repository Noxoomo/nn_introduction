#include "lenet.h"

#include <torch/torch.h>

#include <memory>

namespace experiments::lenet {

// LeNetConv

LeNetConv::LeNetConv() {
    conv1_ = register_module("conv1_", torch::nn::Conv2d(3, 6, 5));
    conv2_ = register_module("conv2_", torch::nn::Conv2d(6, 16, 5));
    conv3_ = register_module("conv3_", torch::nn::Conv2d(16, 16, 1));
//    bn_ = register_module("bn_", torch::nn::BatchNorm(16));
}

torch::Tensor LeNetConv::forward(torch::Tensor x) {
    x = correctDevice(x, *this);
    x = conv1_->forward(x);
    x = torch::max_pool2d(torch::relu(x), 2, 2);
    x = conv2_->forward(x);
//    x = torch::avg_pool2d(x, 2, 2);
//    x = bn_->forward(x);
    x = torch::max_pool2d(torch::relu(x), 2, 2);
    x = conv3_->forward(x);
    return x;
}

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params) {
    return std::make_shared<LeNetConv>();
}

}
