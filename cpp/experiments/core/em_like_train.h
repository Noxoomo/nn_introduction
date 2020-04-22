#pragma once

#include "model.h"
#include "tensor_pair_dataset.h"
#include "loss.h"
#include "optimizer.h"
#include "initializer.h"
#include "params.h"

#include <torch/torch.h>

#include <vector>
#include <util/exception.h>

template <typename TransformType>
class EMLikeTrainer {
public:
    virtual void train(TensorPairDataset& ds, const LossPtr& loss) {
//        VERIFY(model_->classifier()->baseline() == nullptr, "error: baseline unimplemented here");

        prepareDecisionMode(ds, loss);

        int iterations = params_["em_iterations"]["global_iters"];
        for (int i = 0; i < iterations; ++i) {
            std::cout << "EM iteration: " << i << std::endl;

            fireScheduledParamModifiers(i);

            // E step
            optimizeRepresentationModel(ds, loss);
            // M step
            optimizeDecisionModel(ds, loss);

            fireListeners(i);
        }
    }

    virtual experiments::ModelPtr getTrainedModel(TensorPairDataset& ds, const LossPtr& loss) {
        train(ds, loss);
        return model_;
    }

    using IterationListener = std::function<void(uint32_t, experiments::EmModelPtr)>;

    virtual void registerGlobalIterationListener(IterationListener listener) {
        listeners_.push_back(std::move(listener));
    }

    virtual void attachReprEpochEndCallback(experiments::EpochEndCallback cb) {
        reprEpochEndcallbacks_.emplace_back(std::move(cb));
    }

protected:
    EMLikeTrainer(TransformType reprTransform,
            json params,
            experiments::EmModelPtr model)
            : model_(std::move(model))
            , reprTransform_(reprTransform)
            , params_(std::move(params)) {
    }

    virtual experiments::OptimizerPtr getReprOptimizer(const experiments::ModelPtr& reprModel) = 0;

    virtual experiments::OptimizerPtr getDecisionOptimizer(const experiments::ModelPtr& decisionModel) = 0;

    virtual LossPtr makeRepresentationLoss(experiments::ModelPtr model, LossPtr loss) const {
        class ReprLoss : public Loss {
        public:
            ReprLoss(experiments::ModelPtr model, LossPtr loss)
                    : model_(std::move(model))
                    , loss_(std::move(loss)) {

            }

            torch::Tensor value(const torch::Tensor& outputs, const torch::Tensor& targets) const override {
//                std::cout << "getting value from repr loss" << std::endl;
                return loss_->value(model_->forward(outputs), targets);
            }

        private:
            experiments::ModelPtr model_;
            LossPtr loss_;
        };

        return std::make_shared<ReprLoss>(model, loss);
    }

    void fireListeners(uint32_t iteration) {
        std::cout << std::endl;

        model_->eval();
        for (auto& listener : listeners_) {
            listener(iteration, model_);
        }

        std::cout << std::endl;
    }

protected:
    experiments::EmModelPtr model_;
    TransformType reprTransform_;

    json params_;

    std::vector<IterationListener> listeners_;
    std::vector<experiments::EpochEndCallback> reprEpochEndcallbacks_;

private:
    void optimizeRepresentationModel(TensorPairDataset& ds, const LossPtr& loss) {
        if (params_["em_iterations"]["e_iters"] == 0) {
            return;
        }

        auto representationsModel = model_->eStepModel();
        auto decisionModel = model_->mStepModel();

        representationsModel->train(true);
        decisionModel->train(false);

        std::cout << "optimizing representation model" << std::endl;

        LossPtr representationLoss = makeRepresentationLoss(decisionModel, loss);
        auto representationOptimizer = getReprOptimizer(representationsModel);
        representationOptimizer->train(ds, representationLoss, representationsModel);
    }

    void optimizeDecisionModel(TensorPairDataset& ds, const LossPtr& loss) {
        if (params_["em_iterations"]["m_iters"] == 0) {
            return;
        }

        auto representationsModel = model_->eStepModel();
        auto decisionModel = model_->mStepModel();

        representationsModel->train(false);
        decisionModel->train(true);

        std::cout << "getting representations" << std::endl;

        int batchSize = params_[BatchSizeKey];

        auto mds = ds.map(reprTransform_);
        auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(batchSize));
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

        std::cout << "optimizing decision model" << std::endl;

        auto decisionFuncOptimizer = getDecisionOptimizer(decisionModel);
        decisionFuncOptimizer->train(repr, targets, loss, decisionModel);
    }

    void prepareDecisionMode(TensorPairDataset& ds, const LossPtr& loss) {
        // TODO skip for now
    }

    void fireScheduledParamModifiers(int iter) {
        const std::vector<json> paramModifiers = params_[ScheduledParamModifiersKey];

        for (const json& modifier : paramModifiers) {
            std::string field = modifier[FieldKey];
            std::vector<int> iters = modifier[ItersKey];
            std::string type = modifier["type"];

            std::cout << field << std::endl;
            if (type == "double") {
                std::vector<double> values = modifier[ValuesKey];
                for (int i = 0; i < iters.size(); ++i) {
                    if (iters[i] == iter) {
                        std::cout << "changing " << field << " on iter " << iter << " to " << values[i] << std::endl;
                        setField(params_, field, values[i]);
                        break;
                    }
                }
            } else {
                std::vector<int> values = modifier[ValuesKey];
                for (int i = 0; i < iters.size(); ++i) {
                    if (iters[i] == iter) {
                        std::cout << "changing " << field << " on iter " << iter << " to " << values[i] << std::endl;
                        setField(params_, field, values[i]);
                        break;
                    }
                }
            }
        }
    }

};
