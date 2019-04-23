#pragma once

#include "model.h"

#include <torch/torch.h>

#include <vector>

enum class ResNetConfiguration {
    ResNet18,
    ResNet34,
};

// TODO(equivalence1) ResNet and Vgg don't support save/restore features, since
// they store layers using vectors, functions, etc. I need to figure out how to
// properly register submodules to be able to restore models. (Sequential might help).
// Also might be some trouble with private/public fields.


// BasicBlock

class BasicBlock : public experiments::Model {
public:
    BasicBlock(int inChannels, int outChannels, int stride = 1);

    torch::Tensor forward(torch::Tensor x) override;

    ~BasicBlock() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::Conv2d conv2_{nullptr};
    torch::nn::BatchNorm bn1_{nullptr};
    torch::nn::BatchNorm bn2_{nullptr};

    torch::nn::Sequential shortcut_{nullptr};
};

// ResNetConv

class ResNetConv : public experiments::Model {
public:
    ResNetConv(std::vector<int> numBlocks,
               const std::function<experiments::ModelPtr(int, int, int)>& blocksBuilder);

    torch::Tensor forward(torch::Tensor x) override;

    ~ResNetConv() override = default;

private:
    torch::nn::Conv2d conv1_{nullptr};
    torch::nn::BatchNorm bn1_{nullptr};

    std::vector<experiments::ModelPtr> blocks_;
};

// ResNetClassifier

class ResNetClassifier : public experiments::Model {
public:
    explicit ResNetClassifier(int expansion);

    torch::Tensor forward(torch::Tensor x) override;

    ~ResNetClassifier() override = default;

private:
    torch::nn::Linear fc1_{nullptr};
};

// ResNet

class ResNet : public experiments::ConvModel {
public:
    explicit ResNet(ResNetConfiguration cfg, std::shared_ptr<experiments::Model> classifier = std::make_shared<ResNetClassifier>(1));

    torch::Tensor forward(torch::Tensor x) override;

    experiments::ModelPtr conv() override;

    experiments::ModelPtr classifier() override;

    ~ResNet() override = default;

private:
    void init(std::vector<int> nBlocks, std::shared_ptr<experiments::Model> classifier);

private:
    std::shared_ptr<ResNetConv> conv_{nullptr};
    experiments::ModelPtr classifier_{nullptr};
};
