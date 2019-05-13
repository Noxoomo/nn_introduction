#include "common.h"
#include "common_em.h"
#include "catboost_nn.h"

#include <experiments/core/networks/lenet.h>
#include <experiments//datasets/cifar10/cifar10_reader.h>
#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/em_like_train.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>
#include <experiments/core/polynom_model.h>

using namespace  experiments;
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
    const std::string& params = "../../../../resources/cifar10/cifar-10-batches-bin";
    auto dataset = cifar10::read_dataset(path);

    // Init model
    CatBoostNNConfig catBoostNnConfig;
    catBoostNnConfig.batchSize = 128;
    catBoostNnConfig.lambda_ = 1;
    catBoostNnConfig.sgdStep_ = 0.1;
    catBoostNnConfig.representationsIterations = 10;
    catBoostNnConfig.globalIterationsCount = 1000;

    catBoostNnConfig.catboostParamsFile = "../../../../cpp/apps/cifar_networks/lenet_params/catboost_params_gpu.json";
    catBoostNnConfig.catboostInitParamsFile = "../../../../cpp/apps/cifar_networks/lenet_params/catboost_params_init.json";
    catBoostNnConfig.catboostFinalParamsFile = "../../../../cpp/apps/cifar_networks/lenet_params/catboost_params_final.json";

    PolynomPtr polynom = std::make_shared<Polynom>();
    polynom->Lambda_ = catBoostNnConfig.lambda_;
    {
        Monom emptyMonom;
        emptyMonom.Structure_.Splits.push_back({0, 0});
        const auto outDim = 10;
        emptyMonom.Values_.resize(outDim);
        polynom->Ensemble_.push_back(emptyMonom);
    }

    auto lenet = std::make_shared<LeNet>(makeClassifierWithBaseline<PolynomModel>(
        makeCifarLinearClassifier(16 * 5 * 5),
//        makeCifarBias(),
        polynom));
//    auto lenet = std::make_shared<LeNet>(makeClassifier<PolynomModel>(
//        polynom));
    lenet->to(device);

    torch::setNumThreads(16);

    CatBoostNN nnTrainer(catBoostNnConfig,
                         lenet,
                         device,
                         makeClassifier<LinearCifarClassifier>(16 *5 *5));

    // Attach Listener



    auto mds = dataset.second.map(getDefaultCifar10TestTransform());
    nnTrainer.registerGlobalIterationListener([&](uint32_t epoch, ModelPtr model) {
        AccuracyCalcer<decltype(mds)>(device, catBoostNnConfig, mds, nnTrainer)(epoch, model);
    });


    nnTrainer.registerGlobalIterationListener([&](uint32_t epoch, ModelPtr model) {
        if (epoch % 2 != 0) {
            std::cout << "--------===============CATBOOST learn + test start ====================---------------  "
                      << std::endl;
            auto learn = nnTrainer.applyConvLayers(dataset.first.map(getCifar10TrainFinalCatboostTransform()));
            auto test = nnTrainer.applyConvLayers(dataset.second.map(getDefaultCifar10TestTransform()));
            nnTrainer.trainFinalDecision(learn, test);
            std::cout << "--------===============CATBOOST learn + test finish ====================---------------  "
                      << std::endl;
        }
    });
    // Train

    auto loss = std::make_shared<CrossEntropyLoss>();
    nnTrainer.train(dataset.first, loss);

    // Eval model

    auto acc = evalModelTestAccEval(dataset.second,
                                    lenet,
                                    device,
                                    getDefaultCifar10TestTransform());

    std::cout << "LeNet EM test accuracy: " << std::setprecision(2)
              << acc << "%" << std::endl;
}
