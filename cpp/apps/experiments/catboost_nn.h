#include "common.h"
#include <experiments/datasets/cifar10/cifar10_reader.h>
#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/em_like_train.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

struct CatBoostNNConfig {
    uint32_t globalIterationsCount = 500;
    uint32_t representationsIterations = 3;
    double dropOut_ = 0;

    int batchSize = 256;
    double lambda_ = 1.0;
    std::string catboostParamsFile = "catboost_params.json";
    std::string catboostInitParamsFile = "catboost_params.json";
    std::string catboostFinalParamsFile = "catboost_final_params.json";

    double sgdStep_ = 0.001;

};

class CatBoostNN : public EMLikeTrainer<decltype(getDefaultCifar10TrainTransform())> {
public:
    using ConvModelPtr = std::shared_ptr<experiments::ConvModel>;

    CatBoostNN(const CatBoostNNConfig& opts,
        ConvModelPtr model,
        torch::DeviceType device,
        experiments::ClassifierPtr init = nullptr)
            : EMLikeTrainer(getDefaultCifar10TrainTransform(), opts.globalIterationsCount, model)
            , opts_(opts)
            , device_(device)
            , initClassifier_(init) {

    }

    template <class Ds>
    TensorPairDataset applyConvLayers(const Ds& ds) {
        auto representationsModel = model_->conv();
        representationsModel->eval();

        auto dloader = torch::data::make_data_loader(ds, torch::data::DataLoaderOptions(256));
        auto device = representationsModel->parameters().data()->device();
        std::vector<torch::Tensor> reprList;
        std::vector<torch::Tensor> targetsList;

        for (auto& batch : *dloader) {
            auto res = representationsModel->forward(batch.data.to(device));
            auto target = batch.target.to(device);
            reprList.push_back(res);
            targetsList.push_back(target);
        }

        auto repr = torch::cat(reprList, 0);
        auto targets = torch::cat(targetsList, 0);
        return TensorPairDataset(repr, targets);
    }

    void setLambda(double lambda);

    experiments::ModelPtr trainFinalDecision(const TensorPairDataset& learn, const TensorPairDataset& test);

    void train(TensorPairDataset& ds, const LossPtr& loss) override;

    experiments::ModelPtr getTrainedModel(TensorPairDataset& ds, const LossPtr& loss) override;

protected:
    void trainDecision(TensorPairDataset& ds, const LossPtr& loss, double step = -1, double lambda = 1);
    void trainRepr(TensorPairDataset& ds, const LossPtr& loss);
    void initialTrainRepr(TensorPairDataset& ds, const LossPtr& loss);
protected:
    experiments::OptimizerPtr getReprOptimizer(const experiments::ModelPtr& reprModel) override;

    experiments::OptimizerPtr getDecisionOptimizer(const experiments::ModelPtr& decisionModel) override;
    experiments::OptimizerPtr getDecisionOptimizerWithLearningRate(const experiments::ModelPtr& decisionModel, double learning_rate = -1);
private:
    const CatBoostNNConfig& opts_;
    torch::DeviceType device_;
    experiments::ClassifierPtr initClassifier_;
    int64_t seed_ = 0;
    bool Init_ = true;
    double lr_ = 0;
    double lambdaMult_ = 1.0;
    int iter_ = 1;
};




template <class DataSet>
class AccuracyCalcer {
public:
    AccuracyCalcer(c10::DeviceType device, const CatBoostNNConfig& opts, DataSet& mds, CatBoostNN& nnTrainer)
        : device_(device), opts_(opts), mds_(mds), nnTrainer_(nnTrainer) {

    }

    void operator()(uint32_t epoch, experiments::ModelPtr model) {
        model->to(device_);
        model->eval();

        auto dloader = torch::data::make_data_loader(mds_, torch::data::DataLoaderOptions(256));
        int rightAnswersCnt = 0;
        int rightAnswersExactCnt = 0;
        double total = 0;

        for (auto& batch : *dloader) {
            auto data = batch.data;
            data = data.to(device_);
            torch::Tensor target = batch.target;

            torch::Tensor prediction = model->forward(data);
            prediction = torch::argmax(prediction, 1);

            nnTrainer_.setLambda(100000);
            torch::Tensor predictionExact = model->forward(data);
            predictionExact = torch::argmax(predictionExact, 1);
            nnTrainer_.setLambda(opts_.lambda_);

            prediction = prediction.to(torch::kCPU);
            predictionExact = predictionExact.to(torch::kCPU);

            auto targetAccessor = target.accessor<int64_t, 1>();
            auto predictionsAccessor = prediction.accessor<int64_t, 1>();
            auto predictionsExactAccessor = predictionExact.accessor<int64_t, 1>();
            int size = target.size(0);

            for (int i = 0; i < size; ++i) {
                const int targetClass = targetAccessor[i];
                const int predictionClass = predictionsAccessor[i];
                const int predictionExactClass = predictionsExactAccessor[i];
                if (targetClass == predictionClass) {
                    rightAnswersCnt++;
                }
                if (targetClass == predictionExactClass) {
                    rightAnswersExactCnt++;
                }
                ++total;
            }
        }

        std::cout << "Test accuracy: " <<  rightAnswersCnt * 100.0f / total << std::endl;
        std::cout << "Test accuracy (lambda = 100000): " <<  rightAnswersExactCnt * 100.0f / total << std::endl;
    }
private:
    torch::DeviceType device_;
    const CatBoostNNConfig& opts_;
    DataSet& mds_;
    CatBoostNN& nnTrainer_;
};
