#include "common.h"
#include <cifar_nn/cifar10_reader.h>
#include <cifar_nn/optimizer.h>
#include <cifar_nn/cross_entropy_loss.h>
#include <cifar_nn/em_like_train.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

struct CatBoostNNConfig {
    uint32_t globalIterationsCount = 500;
    uint32_t representationsIterations = 3;

    int batchSize = 256;
    double lambda_ = 1.0;
    std::string catboostParamsFile = "catboost_params.json";

    double adamStep = 0.0005;

};

class CatBoostNN : public EMLikeTrainer<decltype(getDefaultCifar10TrainTransform())> {
public:
    using ConvModelPtr = std::shared_ptr<experiments::ConvModel>;

    CatBoostNN(CatBoostNNConfig opts,
        ConvModelPtr model,
        torch::DeviceType device)
            : EMLikeTrainer(getDefaultCifar10TrainTransform(), opts.globalIterationsCount)
            , opts_(opts)
            , model_(std::move(model))
            , device_(device) {

        initializer_ = std::make_shared<NoopInitializer>();

        representationsModel = model_->conv();
        decisionModel = model_->classifier();
    }

    experiments::ModelPtr getTrainedModel(TensorPairDataset& ds, const LossPtr& loss) override;

protected:
    experiments::OptimizerPtr getReprOptimizer(const experiments::ModelPtr& reprModel) override;

    experiments::OptimizerPtr getDecisionOptimizer(const experiments::ModelPtr& decisionModel) override;
private:
    CatBoostNNConfig opts_;
    ConvModelPtr model_;
    torch::DeviceType device_;
    int64_t seed_ = 0;
};
