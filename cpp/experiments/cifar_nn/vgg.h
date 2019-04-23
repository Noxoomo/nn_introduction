#pragma once

#include "model.h"

#include <torch/torch.h>

#include <functional>
#include <vector>

enum class VggConfiguration {
    Vgg16,
};

// VggConv

// See https://arxiv.org/pdf/1409.1556.pdf
//
// We'll be comparing our accuracy with https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
// so in each configuration we use the same conv/maxpool/batchnorm layers as they do.
class VggConv : public experiments::Model {
public:
    VggConv() = default;

    torch::Tensor forward(torch::Tensor x) override = 0;

    ~VggConv() override = default;

protected:
    std::vector<std::function<torch::Tensor(torch::Tensor)>> layers_;
};

// Vgg16Conv

class Vgg16Conv : public VggConv {
public:
    Vgg16Conv();

    torch::Tensor forward(torch::Tensor x) override;

    ~Vgg16Conv() override = default;
};

// VggClassifier

class VggClassifier : public experiments::Model {
public:
    VggClassifier();

    torch::Tensor forward(torch::Tensor x) override;

    ~VggClassifier() override = default;

private:
    torch::nn::Linear fc1_{nullptr};
};

// Vgg

class Vgg : public experiments::ConvModel {
public:
    explicit Vgg(VggConfiguration cfg, std::shared_ptr<experiments::Model> classifier = std::make_shared<VggClassifier>());

    torch::Tensor forward(torch::Tensor x) override;

    experiments::ModelPtr conv() override;

    experiments::ModelPtr classifier() override;

    ~Vgg() override = default;

private:
    std::shared_ptr<VggConv> conv_;
    std::shared_ptr<experiments::Model> classifier_;
};
