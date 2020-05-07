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
              const json& params,
              TensorPairDataset valDs);

private:
    experiments::OptimizerPtr getReprOptimizer(const experiments::ModelPtr& reprModel) override;
    experiments::OptimizerPtr getDecisionOptimizer(const experiments::ModelPtr& decisionModel) override;
    experiments::OptimizerPtr getLinearPolynomOptimizer(const experiments::ModelPtr& decisionModel);
    experiments::OptimizerPtr getCatboostPolynomOptimizer(const std::shared_ptr<PolynomModel>& model);

    void pretrainReprModel(TensorPairDataset& ds, const LossPtr& loss) override;

    // TODO move this optimizers out of here

    class LinearTreesOptimizer : public experiments::Optimizer {
    public:
        explicit LinearTreesOptimizer(const LinearTreesBoosterOptions& opts)
                : opts_(opts) {

        }

        void train(TensorPairDataset& tpds, LossPtr loss, experiments::ModelPtr model) const override;

        void train(TensorPairDataset& trainTpds, TensorPairDataset& valTpds, LossPtr loss, experiments::ModelPtr model) const override;

    private:
        LinearTreesBoosterOptions opts_;
    };

    class CatBoostOptimizer : public experiments::Optimizer {
    public:
        CatBoostOptimizer(std::string catboostOptions,
                          uint64_t seed,
                          double lambda,
                          double dropOut,
                          Monom::MonomType monomType,
                          TensorPairDataset valDs)
                : catBoostOptions_(std::move(catboostOptions))
                , seed_(seed)
                , lambda_(lambda)
                , drouput_(dropOut)
                , monomType_(monomType)
                , valDs_(std::move(valDs)) {

        }

        void train(TensorPairDataset& trainDs,
                   TensorPairDataset& validationDs,
                   LossPtr loss,
                   experiments::ModelPtr model) const override;

        void train(TensorPairDataset &trainDs,
                   LossPtr loss,
                   experiments::ModelPtr model) const override;

    private:
        std::string catBoostOptions_;
        uint64_t seed_ = 0;
        double lambda_ = 1.0;
        double drouput_ = 0.0;
        Monom::MonomType monomType_;
        mutable TensorPairDataset valDs_; // TODO this shouldn't be here, but I need a quick fix
    };

private:
    TensorPairDataset valDs_; // TODO this shouldn't be here, but I need a quick fix

};
