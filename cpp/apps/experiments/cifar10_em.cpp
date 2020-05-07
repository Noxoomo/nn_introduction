#include "cifar10_em.h"

#include <experiments/core/optimizer.h>
#include <experiments/core/cross_entropy_loss.h>
#include <experiments/core/em_like_train.h>
#include <experiments/core/params.h>
#include <core/polynom_model.h>
#include <models/polynom/linear_monom.h>
#include <methods/linear_trees_booster.h>
#include <catboost_wrapper.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

// CommonEm

Cifar10EM::Cifar10EM(experiments::EmModelPtr model,
                     const json& params,
                     TensorPairDataset valDs)
        : EMLikeTrainer(getDefaultCifar10TrainTransform(),
                        params,
                        std::move(model))
        , valDs_(std::move(valDs)) {

}

experiments::OptimizerPtr Cifar10EM::getReprOptimizer(const experiments::ModelPtr& reprModel) {
    auto transform = getDefaultCifar10TrainTransform();
    using TransT = decltype(transform);

    experiments::OptimizerArgs<TransT> args(transform, params_["em_iterations"]["e_iters"]);

    // TODO don't hardcode consts, construct from params
    torch::optim::SGDOptions opt(params_[SgdStepSizeKey]);
    opt.momentum_ = 0.9;
    opt.weight_decay_ = 5e-4;
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

experiments::OptimizerPtr Cifar10EM::getDecisionOptimizer(const experiments::ModelPtr& decisionModel) {
    auto polynomModel = std::dynamic_pointer_cast<PolynomModel>(decisionModel);

    if (polynomModel) {
        if (polynomModel->getMonomType() == Monom::MonomType::LinearMonom) {
            return getLinearPolynomOptimizer(decisionModel);
        } else {
            return getCatboostPolynomOptimizer(polynomModel);
        }
    } else {
        return std::make_shared<experiments::NoopOptimizer>();
    }
}

experiments::OptimizerPtr Cifar10EM::getLinearPolynomOptimizer(const experiments::ModelPtr& decisionModel) {
    LinearTreesBoosterOptions opts = LinearTreesBoosterOptions::fromJson(params_["decision_model_optimizer"]);
    return std::make_shared<LinearTreesOptimizer>(opts);
}

experiments::OptimizerPtr Cifar10EM::getCatboostPolynomOptimizer(const std::shared_ptr<PolynomModel>& model) {
    return std::make_shared<CatBoostOptimizer>(params_["catboost_params"].dump(),
            42,
            params_["model"]["m_model"]["lambda"],
            0.0f,
            model->getMonomType(),
            getRepr(valDs_));
}

void Cifar10EM::pretrainReprModel(TensorPairDataset& ds, const LossPtr& loss) {
    using namespace experiments;
    auto model = std::make_shared<ConvModel>(model_->eStepModel(),
            std::make_shared<Classifier>(std::make_shared<MLP>(std::vector<int>({400, 10}))));
    auto optim = getReprOptimizer(model);
    optim->train(ds, loss, model);
}

void Cifar10EM::LinearTreesOptimizer::train(TensorPairDataset& tpds, LossPtr loss, experiments::ModelPtr model) const {
    auto polynomModel = std::dynamic_pointer_cast<PolynomModel>(model);
    if (!polynomModel) {
        throw std::runtime_error("model should be polynom");
    }

    auto flatData = tpds.data().to(torch::kCPU).view({(long)tpds.size().value(), -1}).contiguous();
    Mx dsdata(Vec(flatData), flatData.sizes()[0], flatData.sizes()[1]);
    DataSet ds(dsdata, Vec(tpds.targets().to(torch::kCPU).to(torch::kFloat).contiguous()));

    LinearTreesBooster booster(opts_);
    auto ensemble = booster.fit(ds);
    auto polynom = std::make_shared<Polynom>(LinearTreesToPolynom(*std::dynamic_pointer_cast<Ensemble>(ensemble)));
    polynomModel->reset(polynom);
}

void Cifar10EM::LinearTreesOptimizer::train(TensorPairDataset& trainTpds, TensorPairDataset& valTpds,
        LossPtr loss, experiments::ModelPtr model) const {
    auto polynomModel = std::dynamic_pointer_cast<PolynomModel>(model);
    if (!polynomModel) {
        throw std::runtime_error("model should be polynom");
    }

    auto flatTrainData = trainTpds.data().to(torch::kCPU).view({(long)trainTpds.size().value(), -1}).contiguous();
    Mx trainDsData(Vec(flatTrainData), flatTrainData.sizes()[0], flatTrainData.sizes()[1]);
    DataSet trainDs(trainDsData, Vec(trainTpds.targets().to(torch::kCPU).to(torch::kFloat).contiguous()));

    auto flatValData = valTpds.data().to(torch::kCPU).view({(long)valTpds.size().value(), -1}).contiguous();
    Mx valDsData(Vec(flatValData), flatValData.sizes()[0], flatValData.sizes()[1]);
    DataSet valDs(valDsData, Vec(valTpds.targets().to(torch::kCPU).to(torch::kFloat).contiguous()));

    LinearTreesBooster booster(opts_);
    auto ensemble = booster.fit(trainDs, valDs);
    auto polynom = std::make_shared<Polynom>(LinearTreesToPolynom(*std::dynamic_pointer_cast<Ensemble>(ensemble)));
    polynomModel->reset(polynom);
}

inline TDataSet MakePool(int fCount,
                         int samplesCount,
                         const float *features,
                         const float *labels,
                         const float *weights = nullptr,
                         const float *baseline = nullptr,
                         int baselineDim = 0) {
    TDataSet pool;
    pool.Features = features;
    pool.Labels = labels;
    pool.FeaturesCount = fCount;
    pool.SamplesCount = samplesCount;
    pool.Weights = weights;
    pool.Baseline = baseline;
    pool.BaselineDim = baselineDim;
    return pool;
}

//static TensorPairDataset binarizeDs(TensorPairDataset &ds, double border) {
//    auto newData = torch::gt(ds.data(), border).to(torch::kFloat32).view({ds.data().size(0), -1});
//    return TensorPairDataset(newData, ds.targets());
//}

void Cifar10EM::CatBoostOptimizer::train(TensorPairDataset& trainDs,
           TensorPairDataset& validationDs,
           LossPtr loss,
           experiments::ModelPtr decisionModel) const {
    auto polynomModel = std::dynamic_pointer_cast<PolynomModel>(decisionModel);

    int nTrainRows = trainDs.size().value();
    int nValidationRows = validationDs.size().value();
    auto trainData = trainDs.data().reshape({nTrainRows, -1}).to(torch::kCPU).t().contiguous();
    auto validationData = validationDs.data().reshape({nValidationRows, -1}).to(torch::kCPU).t().contiguous();

    auto trainTargets = trainDs.targets().to(torch::kCPU, torch::kFloat32).contiguous();
    auto validationTargets = validationDs.targets().to(torch::kCPU, torch::kFloat32).contiguous();

    const int64_t featuresCount = trainData.size(0);

    std::cout << "train data shape: " << trainData.sizes() << std::endl;
    std::cout << "validation data shape: " << validationData.sizes() << std::endl;

    // TODO dropout

    TDataSet trainPool = MakePool(
            featuresCount,
            nTrainRows,
            trainData.data_ptr<float>(),
            trainTargets.data_ptr<float>(),
            nullptr,
            nullptr,
            0);
    TDataSet validationPool = MakePool(
            featuresCount,
            nValidationRows,
            validationData.data_ptr<float>(),
            validationTargets.data_ptr<float>(),
            nullptr,
            nullptr,
            0);

    std::cout << "Training catboost with options: " << catBoostOptions_ << std::endl;

    auto catboost = Train(trainPool, validationPool, catBoostOptions_);
    std::cout << "CatBoost was trained " << std::endl;

    auto polynom = std::make_shared<Polynom>(monomType_, PolynomBuilder().AddEnsemble(catboost).Build());
    polynom->Lambda_ = lambda_;
    std::cout << "Model size: " << catboost.Trees.size() << std::endl;
    std::cout << "Polynom size: " << polynom->Ensemble_.size() << std::endl;
    std::map<int, int> featureIds;
    int fCount = 0;
    double total = 0;
    for (const auto& monom : polynom->Ensemble_) {
        for (const auto& split : monom->Structure_.Splits) {
            featureIds[split.Feature]++;
            fCount = std::max<int>(fCount, split.Feature);
            ++total;
        }
    }
    std::cout << "Polynom used features: " << featureIds.size() << std::endl;
    for (int k = 0; k <= fCount; ++k) {
        std::cout << featureIds[k] / total << " ";
    }
    std::cout << std::endl << "===============" << std::endl;
    std::cout << std::endl << "Polynom values hist" << std::endl;
    polynom->PrintHistogram();
    std::cout << std::endl << "===============" << std::endl;

    polynomModel->reset(polynom);
}

void Cifar10EM::CatBoostOptimizer::train(TensorPairDataset& trainDs,
                                         LossPtr loss,
                                         experiments::ModelPtr decisionModel) const {
    train(trainDs, valDs_, loss, decisionModel);
}