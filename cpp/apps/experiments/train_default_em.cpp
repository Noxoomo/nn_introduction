#include "common.h"
#include "cifar10_em.h"

#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/params.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

int main(int argc, const char* argv[]) {
    using namespace experiments;

    // Init model

    auto paramsFolder = getParamsFolder(argc, argv);
    auto params = readJson(paramsFolder + "train_em_params.json");

    auto device = getDevice(params[DeviceKey]);
    int batchSize = params[BatchSizeKey];

    auto model = std::dynamic_pointer_cast<EmModel>(experiments::createModel(params["model"]));
    if (!model) {
        throw std::runtime_error("Provided model is not EM model");
    }

    // Read dataset
    auto dataset = readDataset(params[DatasetKey]);

    // Init trainer

    Cifar10EM emTrainer(model, params);

    // Attach Listeners

    auto mds = dataset.second.map(getDefaultCifar10TestTransform());
    emTrainer.attachReprEpochEndCallback([&](int epoch, Model& model) {
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

    if (params.contains("checkpoint_file")) {
        emTrainer.registerGlobalIterationListener([&](int32_t globalIt, EmModelPtr model) {
            const std::string& path = params["checkpoint_file"];
            std::cout << "Saving e model to '" << path << "'" << std::endl;
            torch::save(model->eStepModel(), path);
        });
    }

    // Train

    auto loss = std::make_shared<CrossEntropyLoss>();
    emTrainer.train(dataset.first, loss);

    // Eval model

    auto acc = evalModelTestAccEval(dataset.second,
                                    model,
                                    getDefaultCifar10TestTransform());

    std::cout << "Test accuracy: " << std::setprecision(2)
              << acc << "%" << std::endl;
}
