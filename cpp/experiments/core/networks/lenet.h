#pragma once

#include <experiments/core/model.h>
#include <util/json.h>

#include <torch/torch.h>

namespace experiments::lenet {

// LeNetConv

class LeNetConv : public Model {
public:
    LeNetConv();

    torch::Tensor forward(torch::Tensor x) override;

    ~LeNetConv() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
//    torch::nn::BatchNorm bn_{nullptr};
};

ModelPtr createConvLayers(const std::vector<int>& inputShape, const json& params);

}
