#include "common.h"
#include "cifar10_em.h"
#include "catboost_nn.h"

#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/em_like_train.h>
#include <experiments/core/polynom_model.h>
#include <experiments/core/params.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>
#include <set>

int main(int argc, const char* argv[]) {
    using namespace experiments;

    torch::manual_seed(0);

    // Load params

    auto paramsFolder = getParamsFolder(argc, argv);
    auto params = readJson(paramsFolder + "train_with_caboost_params.json");

    std::cout << "Running with parameters: {{{" << std::endl;
    std::cout << params.dump(4) << std::endl;
    std::cout << "}}}\n\n\n" << std::endl;

    // Read dataset

    auto dataset = readDataset(params[DatasetKey]);

    // Init model

    const json& convParams = params[ModelKey][ConvKey];
    const json& classParams = params[ModelKey][ClassifierKey];

    auto conv = createConvLayers({}, convParams);
    auto classifier = createClassifier(2, classParams);

    auto model = std::make_shared<EmModel>(conv, classifier);

    //torch::setNumThreads(16);

    CatBoostNN nnTrainer(params,
                         model,
                         dataset.second);
//                         createClassifier(2, classParams));

    // Attach Listener

    auto mds = dataset.second.map(getDefaultCifar10TestTransform());
    nnTrainer.registerGlobalIterationListener([&](uint32_t epoch, ModelPtr model) {
        // TODO params might have changed
        AccuracyCalcer<decltype(mds)>(params, mds, nnTrainer)(epoch, model);
    });


    nnTrainer.registerGlobalIterationListener([&](uint32_t epoch, ModelPtr model) {
        if (epoch % 2 != 0) {
            std::cout << "--------===============CATBOOST learn + test start ====================---------------  "
                      << std::endl;
            std::cout << "skipping trainFinalDecision" << std::endl;
//            auto learn = nnTrainer.applyConvLayers(dataset.first.map(getCifar10TrainFinalCatboostTransform()));
//            auto test = nnTrainer.applyConvLayers(dataset.second.map(getDefaultCifar10TestTransform()));
//            nnTrainer.trainFinalDecision(learn, test);
            std::cout << "--------===============CATBOOST learn + test finish ====================---------------  "
                      << std::endl;
        }
    });
    // Train

    auto loss = std::make_shared<CrossEntropyLoss>();
    nnTrainer.train(dataset.first, loss);

    // Eval model

    auto acc = evalModelTestAccEval(dataset.second,
                                    model,
                                    getDefaultCifar10TestTransform());

    std::cout << "Test accuracy: " << std::setprecision(2)
              << acc << "%" << std::endl;
}
