#include "common.h"
#include "common_em.h"
#include "catboost_nn.h"

#include <experiments/core/networks/mobile_net_v2.h>
#include <experiments//datasets/cifar10/cifar10_reader.h>
#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/em_like_train.h>
#include <experiments/core/transform.h>
#include <experiments/core/polynom_model.h>

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

    // Read dataset

    const std::string& path = "../../../../resources/cifar10/cifar-10-batches-bin";
    auto dataset = cifar10::read_dataset(path);

    // Init model

    CatBoostNNConfig catBoostNnConfig;
    catBoostNnConfig.batchSize = 128;
    catBoostNnConfig.lambda_ = 1;
    catBoostNnConfig.sgdStep_ = 0.01;
    catBoostNnConfig.representationsIterations = 10;
    catBoostNnConfig.catboostParamsFile = "../../../../cpp/apps/cifar_networks/catboost_params_gpu.json";
    catBoostNnConfig.catboostInitParamsFile = "../../../../cpp/apps/cifar_networks/catboost_params_init.json";
    catBoostNnConfig.catboostFinalParamsFile = "../../../../cpp/apps/cifar_networks/catboost_params_final.json";

    PolynomPtr polynom = std::make_shared<Polynom>();
    polynom->Lambda_ = catBoostNnConfig.lambda_;
    {
        Monom emptyMonom;
        emptyMonom.Structure_ .Splits.push_back({0, 0});
        const auto outDim = 10;
        emptyMonom.Values_.resize(outDim);
        polynom->Ensemble_.push_back(emptyMonom);
    }

    auto mobileNetV2 = std::make_shared<MobileNetV2>(std::make_shared<PolynomModel>(polynom));
    mobileNetV2->to(device);

    CatBoostNN nnTrainer(catBoostNnConfig, mobileNetV2, device);

    // Attach Listeners

    nnTrainer.registerGlobalIterationListener([&](uint32_t epoch, experiments::ModelPtr model) {
        std::cout << "--------===============CATBOOST learn + test start ====================---------------  "  << std::endl;
        auto learn = nnTrainer.applyConvLayers(dataset.first.map(getDefaultCifar10TestTransform()));
        auto test =  nnTrainer.applyConvLayers(dataset.second.map(getDefaultCifar10TestTransform()));
        nnTrainer.trainFinalDecision(learn, test);
        std::cout << "--------===============CATBOOST learn + test finish ====================---------------  "  << std::endl;

    });

    auto mds = dataset.second.map(getDefaultCifar10TestTransform());
    nnTrainer.registerGlobalIterationListener([&](uint32_t epoch, ModelPtr model) {
        nnTrainer.setLambda(10000);
        model->eval();

        auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(128));
        int rightAnswersCnt = 0;

        for (auto& batch : *dloader) {
            auto data = batch.data;
            data = data.to(device);
            torch::Tensor target = batch.target;

            torch::Tensor prediction = model->forward(data);
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
        nnTrainer.setLambda(catBoostNnConfig.lambda_);

        std::cout << "Test accuracy: " <<  rightAnswersCnt * 100.0f / dataset.second.size().value() << std::endl;
    });

    // Train

    auto loss = std::make_shared<CrossEntropyLoss>();
    nnTrainer.train(dataset.first, loss);

    // Eval model

    auto acc = evalModelTestAccEval(dataset.second,
                                    mobileNetV2,
                                    device,
                                    getDefaultCifar10TestTransform());

    std::cout << "MobileNetV2 catboost test accuracy: " << std::setprecision(4)
              << acc << "%" << std::endl;
}
