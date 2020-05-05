#include "common.h"

#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/em_like_train.h>
#include <experiments/core/params.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>
#include <methods/linear_trees_booster.h>
#include <core/polynom_model.h>
#include <models/polynom/linear_monom.h>

// CommonEm

class Cifar10EM final : public EMLikeTrainer<decltype(getDefaultCifar10TrainTransform())> {
public:
    Cifar10EM(experiments::EmModelPtr model,
              const json& params)
            : EMLikeTrainer(getDefaultCifar10TrainTransform(),
                    params,
                    std::move(model)) {
    }

private:
    experiments::OptimizerPtr getReprOptimizer(const experiments::ModelPtr& reprModel) override {
        auto transform = getDefaultCifar10TrainTransform();
        using TransT = decltype(transform);

        experiments::OptimizerArgs<TransT> args(transform, params_["em_iterations"]["e_iters"]);

        // TODO don't hardcode consts, construct from params
        torch::optim::SGDOptions opt(params_[SgdStepSizeKey]);
        opt.momentum_ = 0.9;
        opt.weight_decay_ = 5e-4;
        // TODO dropout
//        auto optim = std::make_shared<torch::optim::Adam>(reprModel->parameters(), opt);
        auto optim = std::make_shared<torch::optim::SGD>(reprModel->parameters(), opt);
        args.torchOptim_ = optim;

        auto* lr = &(optim->options.learning_rate_);
        args.lrPtrGetter_ = [=]() { return lr; };

        auto dloaderOptions = torch::data::DataLoaderOptions(params_[BatchSizeKey]);
        args.dloaderOptions_ = std::move(dloaderOptions);

        auto optimizer = std::make_shared<experiments::DefaultOptimizer<TransT>>(args);
        attachDefaultListeners(optimizer, params_);

        for (const auto& cb : reprEpochEndcallbacks_) {
            experiments::Optimizer::emplaceEpochListener<experiments::EpochEndCallback>(optimizer.get(), cb);
        }

        return optimizer;
    }

    experiments::OptimizerPtr getDecisionOptimizer(const experiments::ModelPtr& decisionModel) override {
        auto polynomModel = std::dynamic_pointer_cast<PolynomModel>(decisionModel);

        if (polynomModel) {
            return getLinearPolynomOptimizer(decisionModel);
        } else {
            return std::make_shared<experiments::NoopOptimizer>();
        }
    }

    class LinearTreesOptimizer : public experiments::Optimizer {
    public:
        explicit LinearTreesOptimizer(const LinearTreesBoosterOptions& opts)
                : opts_(opts) {

        }

        void train(TensorPairDataset& tpds, LossPtr loss, experiments::ModelPtr model) const override {
            auto polynomModel = std::dynamic_pointer_cast<PolynomModel>(model);
            if (!polynomModel) {
                throw std::runtime_error("model should be polynom");
            }

            auto flatData = tpds.data().to(torch::kCPU).view({(long)tpds.size().value(), -1}).contiguous();
            Mx dsdata(Vec(flatData), flatData.sizes()[0], flatData.sizes()[1]);
            DataSet ds(dsdata, Vec(tpds.targets().to(torch::kCPU).contiguous()));

            LinearTreesBooster booster(opts_);
            auto ensemble = booster.fit(ds);
            auto polynom = std::make_shared<Polynom>(LinearTreesToPolynom(*std::dynamic_pointer_cast<Ensemble>(ensemble)));
            polynomModel->reset(polynom);
        }

        void train(TensorPairDataset& trainTpds, TensorPairDataset& valTpds, LossPtr loss, experiments::ModelPtr model) const override {
            auto polynomModel = std::dynamic_pointer_cast<PolynomModel>(model);
            if (!polynomModel) {
                throw std::runtime_error("model should be polynom");
            }

            auto flatTrainData = trainTpds.data().to(torch::kCPU).view({(long)trainTpds.size().value(), -1}).contiguous();
            Mx trainDsData(Vec(flatTrainData), flatTrainData.sizes()[0], flatTrainData.sizes()[1]);
            DataSet trainDs(trainDsData, Vec(trainTpds.targets().to(torch::kCPU).contiguous()));

            auto flatValData = valTpds.data().to(torch::kCPU).view({(long)valTpds.size().value(), -1}).contiguous();
            Mx valDsData(Vec(flatValData), flatValData.sizes()[0], flatValData.sizes()[1]);
            DataSet valDs(valDsData, Vec(valTpds.targets().to(torch::kCPU).contiguous()));

            LinearTreesBooster booster(opts_);
            auto ensemble = booster.fit(trainDs, valDs);
            auto polynom = std::make_shared<Polynom>(LinearTreesToPolynom(*std::dynamic_pointer_cast<Ensemble>(ensemble)));
            polynomModel->reset(polynom);
        }

    private:
        LinearTreesBoosterOptions opts_;
    };

    experiments::OptimizerPtr getLinearPolynomOptimizer(const experiments::ModelPtr& decisionModel) {
        LinearTreesBoosterOptions opts = LinearTreesBoosterOptions::fromJson(params_["decision_model_optimizer"]);
        return std::make_shared<LinearTreesOptimizer>(opts);
    }

};
