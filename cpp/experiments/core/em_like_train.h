#pragma once

#include "model.h"
#include "tensor_pair_dataset.h"
#include "loss.h"
#include "optimizer.h"
#include "initializer.h"

#include <torch/torch.h>

#include <vector>
#include <util/exception.h>

template <typename TransformType>
class EMLikeTrainer {
public:
    virtual void train(TensorPairDataset& ds, const LossPtr& loss) {
        auto representationsModel = model_->conv();
        auto decisionModel = model_->classifier();
        VERIFY(decisionModel->baseline() == nullptr, "error: baseline unimplemented here");

        for (uint32_t i = 0; i < iterations_; ++i) {
            std::cout << "EM iteration: " << i << std::endl;
            representationsModel->train(false);
            decisionModel->train(true);

            std::cout << "    getting representations" << std::endl;

            auto mds = ds.map(reprTransform_);
            auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(1024));
            auto device = representationsModel->parameters().data()->device();


            std::vector<torch::Tensor> reprList;
            std::vector<torch::Tensor> targetsList;

            for (auto& batch : *dloader) {
                auto res = representationsModel->forward(batch.data.to(device)).to(torch::kCPU);
                auto target = batch.target;
                reprList.push_back(res);
                targetsList.push_back(target);
            }

            auto repr = torch::cat(reprList, 0);
            auto targets = torch::cat(targetsList, 0);

            std::cout << "    optimizing decision model" << std::endl;

            auto decisionFuncOptimizer = getDecisionOptimizer(decisionModel);
            decisionFuncOptimizer->train(repr, targets, loss, decisionModel);

            representationsModel->train(true);
            decisionModel->train(false);


            std::cout << "    optimizing representation model" << std::endl;

            LossPtr representationLoss = makeRepresentationLoss(decisionModel, loss);
            auto representationOptimizer = getReprOptimizer(representationsModel);
            representationOptimizer->train(ds, representationLoss, representationsModel);

            fireListeners(i);
        }
    }

    virtual experiments::ModelPtr getTrainedModel(TensorPairDataset& ds, const LossPtr& loss) {
        train(ds, loss);
        return model_;
    }

    virtual LossPtr makeRepresentationLoss(experiments::ModelPtr model, LossPtr loss) const {
        class ReprLoss : public Loss {
        public:
            ReprLoss(experiments::ModelPtr model, LossPtr loss)
                    : model_(std::move(model))
                    , loss_(std::move(loss)) {

            }

            torch::Tensor value(const torch::Tensor& outputs, const torch::Tensor& targets) const override {
                const torch::Tensor& outs = model_->forward(outputs);
                return loss_->value(outs, targets);
            }

        private:
            experiments::ModelPtr model_;
            LossPtr loss_;
        };

        return std::make_shared<ReprLoss>(model, loss);
    }

    using IterationListener = std::function<void(uint32_t, experiments::ModelPtr)>;

    virtual void registerGlobalIterationListener(IterationListener listener) {
        listeners_.push_back(std::move(listener));
    }

protected:
    EMLikeTrainer(TransformType reprTransform, uint32_t iterations, experiments::ConvModelPtr model)
            : model_(model)
            , reprTransform_(reprTransform)
            , iterations_(iterations) {
    }

    virtual experiments::OptimizerPtr getReprOptimizer(const experiments::ModelPtr& reprModel) {
        return {nullptr};
    }

    virtual experiments::OptimizerPtr getDecisionOptimizer(const experiments::ModelPtr& decisionModel) {
        return {nullptr};
    }

protected:
    void fireListeners(uint32_t iteration) {
        std::cout << std::endl;
        model_->eval();
        for (auto& listener : listeners_) {
            listener(iteration, model_);
        }
        model_->train();

        std::cout << std::endl;
    }

protected:
    experiments::ConvModelPtr model_;
    TransformType reprTransform_;


    uint32_t iterations_;

    std::vector<IterationListener> listeners_;
};
