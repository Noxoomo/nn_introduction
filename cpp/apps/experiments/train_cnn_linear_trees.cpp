#include "common.h"

#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/params.h>

#include <util/json.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

int main(int argc, const char* argv[]) {
    using namespace experiments;

    // Init model

    auto paramsFolder = getParamsFolder(argc, argv);
    auto params = readJson(paramsFolder + "train_default_params.json");

    auto device = getDevice(params[DeviceKey]);
    int batchSize = params[BatchSizeKey];

    const json& convParams = params[ModelKey][ConvKey];
    const json& classParams = params[ModelKey][ClassifierKey];

    auto conv = createConvLayers({}, convParams);
    auto classifier = createClassifier(2, classParams);

    auto model = std::make_shared<ConvModel>(conv, classifier);
    model->to(device);

    // Load data
    auto dataset = readDataset(params[DatasetKey]);

    // Create optimizer
    auto optimizer = getDefaultOptimizer(model, params);
    auto loss = std::make_shared<CrossEntropyLoss>();

    // AttachListeners

    std::string modelCheckpoint = params[ModelCheckpointFileKey];
    attachDefaultListeners(optimizer, params);
    auto mds = dataset.second.map(getDefaultCifar10TestTransform());

    experiments::Optimizer::emplaceEpochListener<experiments::EpochEndCallback>(optimizer.get(), [&](int epoch, experiments::Model& model) {
        model.eval();

        auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(batchSize));
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

    optimizer->train(dataset.first, loss, model);

    return 0;
}
