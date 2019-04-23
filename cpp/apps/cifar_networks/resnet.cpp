#include "common.h"
#include <cifar_nn/resnet.h>
#include <cifar_nn/cifar10_reader.h>
#include <cifar_nn/optimizer.h>
#include <cifar_nn/cross_entropy_loss.h>
#include <cifar_nn/transform.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

int main(int argc, char* argv[]) {
    auto device = torch::kCPU;
    if (argc > 1 && std::string(argv[1]) == std::string("CUDA")
            && torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA device for training" << std::endl;
    } else {
        std::cout << "Using CPU device for training" << std::endl;
    }

    // Init model

    auto resnet = std::make_shared<ResNet>(ResNetConfiguration::ResNet18);
    resnet->to(device);

    // Load data

    const std::string& path = "../../../../resources/cifar10/cifar-10-batches-bin";
    auto dataset = cifar::read_dataset(path);

    // Create optimizer

    auto optimizer = getDefaultCifar10Optimizer(400, resnet, device);
    auto loss = std::make_shared<CrossEntropyLoss>();

    // AttachListeners

    attachDefaultListeners(optimizer, 50000 / 128 / 10, "resnet_checkpoint.pt");

    // Train model

    optimizer->train(dataset.first, loss, resnet);

    // Evaluate on test set

    auto acc = evalModelTestAccEval(dataset.second,
            resnet,
            device,
            getDefaultCifar10TestTransform());

    std::cout << "ResNet test accuracy: " << std::setprecision(2)
              << acc << "%" << std::endl;
}
