#include "common.h"

#include <experiments/datasets/cifar10/cifar10_reader.h>
#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/em_like_train.h>
#include <util/json.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>
#include <core/params.h>

class CatBoostNN : public EMLikeTrainer<decltype(getDefaultCifar10TrainTransform())> {
public:
    using ConvModelPtr = std::shared_ptr<experiments::ConvModel>;

    CatBoostNN(json opts,
        ConvModelPtr model,
        TensorPairDataset validationDs,
        bool trainScale = false,
        experiments::ClassifierPtr init = nullptr)
            : EMLikeTrainer(getDefaultCifar10TrainTransform(), opts[NIterationsKey][0], model)
            , opts_(std::move(opts))
            , initClassifier_(init)
            , validationDs_(std::move(validationDs))
            , trainScale_(trainScale){

    }

    template <class Ds>
    TensorPairDataset applyConvLayers(const Ds& ds) {
        auto representationsModel = model_->conv();
        representationsModel->eval();

        auto dloader = torch::data::make_data_loader(ds, torch::data::DataLoaderOptions(256));
        std::vector<torch::Tensor> reprList;
        std::vector<torch::Tensor> targetsList;

        for (auto& batch : *dloader) {
            auto res = representationsModel->forward(batch.data);
            auto target = correctDevice(batch.target, representationsModel);
            reprList.push_back(res);
            targetsList.push_back(target);
        }

        auto repr = torch::cat(reprList, 0);
        auto targets = torch::cat(targetsList, 0);
        return TensorPairDataset(repr, targets);
    }

    void setLambda(double lambda);

    experiments::ModelPtr trainFinalDecision(TensorPairDataset& learn, const TensorPairDataset& test);

    void train(TensorPairDataset& trainDs, const LossPtr& loss) override;

    experiments::ModelPtr getTrainedModel(TensorPairDataset& ds, const LossPtr& loss) override;

protected:
    void trainDecision(TensorPairDataset& ds, const LossPtr& loss);
    void trainRepr(TensorPairDataset& ds, const LossPtr& loss);
    void initialTrainRepr(TensorPairDataset& ds, const LossPtr& loss);
    TensorPairDataset getRepr(TensorPairDataset &ds, const experiments::ModelPtr& reprModel);

protected:
    experiments::OptimizerPtr getReprOptimizer(const experiments::ModelPtr& reprModel) override;

    experiments::OptimizerPtr getDecisionOptimizer(const experiments::ModelPtr& decisionModel) override;

private:
    void fireScheduledParamModifiers(int iter);

private:
    json opts_;
    experiments::ClassifierPtr initClassifier_;
    int64_t seed_ = 0;
    bool Init_ = true;
    double lr_ = 0;
    double lambdaMult_ = 1.0;
    int iter_ = 1;
    TensorPairDataset validationDs_;
    bool trainScale_ = false;
};




template <class DataSet>
class AccuracyCalcer {
public:
    AccuracyCalcer(const json& opts, DataSet& mds, CatBoostNN& nnTrainer)
        : opts_(opts), mds_(mds), nnTrainer_(nnTrainer) {

    }

    void operator()(uint32_t epoch, experiments::ModelPtr model) {
        model->eval();

        auto dloader = torch::data::make_data_loader(mds_, torch::data::DataLoaderOptions(256));
        int rightAnswersCnt = 0;
        int rightAnswersExactCnt = 0;
        double total = 0;

        for (auto& batch : *dloader) {
            auto data = batch.data;
            data = experiments::correctDevice(data, model);
            torch::Tensor target = batch.target;

            torch::Tensor prediction = model->forward(data);
            prediction = torch::argmax(prediction, 1);

            nnTrainer_.setLambda(100000);
            torch::Tensor predictionExact = model->forward(data);
            predictionExact = torch::argmax(predictionExact, 1);
            nnTrainer_.setLambda(opts_[ModelKey][ClassifierKey][ClassifierMainKey][LambdaKey]);

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
    const json& opts_;
    DataSet& mds_;
    CatBoostNN& nnTrainer_;
};
