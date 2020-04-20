#include "common.h"

#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/em_like_train.h>
#include <experiments/core/params.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

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
//        opt.weight_decay_ = 5e-4;
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
//        auto transform = torch::data::transforms::Stack<>();
//        using TransT = decltype(transform);
//
//        experiments::OptimizerArgs<TransT> args(transform, opts_.decisionIterations);
//
//        torch::optim::AdamOptions opt(0.0005);
////        opt.weight_decay_ = 5e-4;
//        auto optim = std::make_shared<torch::optim::Adam>(decisionModel->parameters(), opt);
//        args.torchOptim_ = optim;
//
//        auto lr = &(optim->options.learning_rate_);
//        args.lrPtrGetter_ = [=]() { return lr; };
//
//        const auto batchSize= 256;
//        auto dloaderOptions = torch::data::DataLoaderOptions(batchSize);
//        args.dloaderOptions_ = std::move(dloaderOptions);
//
//        auto optimizer = std::make_shared<experiments::DefaultOptimizer<TransT>>(args);
//        attachDefaultListeners(optimizer, decisionParams_);
        return std::make_shared<experiments::NoopOptimizer>();
    }

};
