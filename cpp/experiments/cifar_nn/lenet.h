#pragma once

#include "model.h"

#include <torch/torch.h>

#include <memory>

// LeNetConv

class LeNetConv : public experiments::Model {
public:
    LeNetConv();

    torch::Tensor forward(torch::Tensor x) override;

    ~LeNetConv() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
};

// LeNetClassifier

class LeNetClassifier : public experiments::Model {
public:
    LeNetClassifier();

    torch::Tensor forward(torch::Tensor x) override;

    ~LeNetClassifier() override = default;

private:
    torch::nn::Linear fc1_{nullptr};
    torch::nn::Linear fc2_{nullptr};
    torch::nn::Linear fc3_{nullptr};
};

// LeNet

class LeNet : public experiments::ConvModel {
public:
    LeNet(std::shared_ptr<experiments::Model> classifier = std::make_shared<LeNetClassifier>());

    torch::Tensor forward(torch::Tensor x) override;

    experiments::ModelPtr conv() override;

    experiments::ModelPtr classifier() override;

    ~LeNet() override = default;

private:
    std::shared_ptr<LeNetConv> conv_{nullptr};
    std::shared_ptr<experiments::Model> classifier_{nullptr};
};

