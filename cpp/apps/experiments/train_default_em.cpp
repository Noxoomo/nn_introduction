#include "common.h"
#include "cifar10_em.h"

#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/params.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>
#include <core/matrix.h>
#include <data/dataset.h>
#include <core/linear_trees_booster.h>

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
    emTrainer.attachReprEpochEndCallback([&](int epoch, experiments::Model& model) {
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

    if (params["model"].contains("checkpoint_file")) {
        emTrainer.registerGlobalIterationListener([&](int32_t globalIt, EmModelPtr model) {
            const std::string& path = params["model"]["checkpoint_file"];
            std::cout << "Saving model to '" << path << "'" << std::endl;
            torch::save(model, path);
        });
    }

    // Train

    auto loss = std::make_shared<CrossEntropyLoss>();
    emTrainer.train(dataset.first, loss);

    auto conv = std::dynamic_pointer_cast<ConvModel>(model->eStepModel())->conv();

    // Eval model

    auto acc = evalModelTestAccEval(dataset.second,
                                    model,
                                    getDefaultCifar10TestTransform());

    std::cout << "Test accuracy: " << std::setprecision(5)
              << acc << "%" << std::endl;

    // Eval with trees

    std::cout << "getting train ds repr" << std::endl;

    auto trainRepr = conv->forward(dataset.first.data()).to(torch::kCPU);
    trainRepr = trainRepr.view({trainRepr.sizes()[0], -1});
    Vec trainReprVec(trainRepr);
    Mx reprTrainDsMx(trainReprVec, trainRepr.sizes()[0], trainRepr.sizes()[1]);
    DataSet trainDs(reprTrainDsMx, Vec(dataset.first.targets().to(torch::kCPU).to(torch::kFloat)));
    trainDs.addBiasColumn();

    std::cout << "getting test ds repr" << std::endl;

    auto testRepr = conv->forward(dataset.second.data()).to(torch::kCPU);
    testRepr = testRepr.view({testRepr.sizes()[0], -1});
    Vec testReprVec(testRepr);
    Mx reprTestDsMx(testReprVec, testRepr.sizes()[0], testRepr.sizes()[1]);
    DataSet testDs(reprTestDsMx, Vec(dataset.second.targets().to(torch::kCPU).to(torch::kFloat)));
    testDs.addBiasColumn();

    std::cout << "parsing options" << std::endl;

    LinearTreesBoosterOptions opts = LinearTreesBoosterOptions::fromJson(params["eval_model"]);
    opts.greedyLinearTreesOpts.biasCol = 0;
    LinearTreesBooster ltBooster(opts);

    std::cout << "fitting ensemble" << std::endl;

    auto ensemble = ltBooster.fit(trainDs, testDs);
}
