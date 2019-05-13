#include "common.h"

#include <experiments/core/networks/mobile_net_v2.h>
#include <experiments//datasets/cifar10/cifar10_reader.h>
#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/transform.h>

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

    using namespace experiments;

    // Init model

    auto mobileNetV2 = std::make_shared<MobileNetV2>();
    mobileNetV2->to(device);

    // Load data

    const std::string& path = "../../../../resources/cifar10/cifar-10-batches-bin";
    auto dataset = cifar10::read_dataset(path);


    // Create optimizer

    auto optimizer = getDefaultCifar10Optimizer(800, mobileNetV2, device);
    auto loss = std::make_shared<CrossEntropyLoss>();

    // AttachListeners

    attachDefaultListeners(optimizer, 50000 / 128 / 10, "mobile_net_v2_checkpoint.pt");

    auto mds = dataset.second.map(getDefaultCifar10TestTransform());
    experiments::Optimizer::emplaceEpochListener<EpochEndCallback>(optimizer.get(), [&](int epoch, experiments::Model& model) {
        model.eval();

        auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(128));
        int rightAnswersCnt = 0;

        for (auto& batch : *dloader) {
            auto data = batch.data;
            data = data.to(device);
            torch::Tensor target = batch.target;

            torch::Tensor prediction = model.forward(data);
            prediction = torch::argmax(prediction, 1);

            prediction = prediction.to(torch::kCPU);

            auto targetAccessor = target.accessor<int64_t, 1>();
            auto predictionsAccessor = prediction.accessor<int64_t, 1>();
            int size = target.size(0);

            for (int i = 0; i < size; ++i) {
                const int targetClass = targetAccessor[i];
                const int predictionClass = predictionsAccessor[i];
                if (targetClass == predictionClass) {
                    rightAnswersCnt++;
                }
            }
        }

        std::cout << "Test accuracy: " <<  rightAnswersCnt * 100.0f / dataset.second.size().value() << std::endl;
    });

    // Train model

    optimizer->train(dataset.first, loss, mobileNetV2);

    // Evaluate on test set

    auto acc = evalModelTestAccEval(dataset.second,
            mobileNetV2,
            device,
            getDefaultCifar10TestTransform());

    std::cout << "MobileNetV2 test accuracy: " << std::setprecision(4)
              << acc << "%" << std::endl;
}
