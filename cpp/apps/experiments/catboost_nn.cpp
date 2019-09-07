#include "catboost_nn.h"

#include <experiments/core/polynom_model.h>
#include <models/polynom/polynom.h>
#include <core/vec.h>
#include <util/io.h>
#include <util/string_utils.h>

#include <catboost_wrapper.h>

#include <utility>
#include <random>
#include <stdexcept>
#include <memory>

experiments::ModelPtr CatBoostNN::getTrainedModel(TensorPairDataset& ds, const LossPtr& loss) {
    train(ds, loss);
    return model_;
}

inline TDataSet MakePool(int fCount,
                         int samplesCount,
                         const float *features,
                         const float *labels,
                         const float *weights = nullptr,
                         const float *baseline = nullptr,
                         int baselineDim = 0
                        ) {
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


class LrLinearDecayOptimizerListener : public experiments::OptimizerEpochListener {
public:
  LrLinearDecayOptimizerListener(double from, double to, int lastEpoch)
  : from_(from)
  , to_(to)
  , lastEpoch_(lastEpoch) {

  }

  void epochReset() override {

  }

  void onEpoch(int epoch, double* lr, experiments::ModelPtr model) override {
    *lr = from_ + (to_ - from_) * epoch / lastEpoch_;
  }

private:
  double from_;
  double to_;
  int lastEpoch_;
};


static void attachReprListeners(const experiments::OptimizerPtr& optimizer,
                            int nBatchesReport, int epochs, double startLr, double endLr) {
  // report 10 times per epoch
  auto brListener = std::make_shared<experiments::BatchReportOptimizerListener>(nBatchesReport);
  optimizer->registerListener(brListener);

  auto epochReportOptimizerListener = std::make_shared<experiments::EpochReportOptimizerListener>();
  optimizer->registerListener(epochReportOptimizerListener);

  auto lrDecayListener = std::make_shared<LrLinearDecayOptimizerListener>(startLr, endLr, epochs);
  optimizer->registerListener(lrDecayListener);
}



experiments::OptimizerPtr CatBoostNN::getReprOptimizer(const experiments::ModelPtr& model) {
    auto transform = getDefaultCifar10TrainTransform();
    using TransT = decltype(transform);

    experiments::OptimizerArgs<TransT> args(transform, iter_);
//
    torch::optim::SGDOptions opt(lr_);
    opt.momentum_ = 0.9;
//    torch::optim::AdamOptions opt(lr);
    opt.weight_decay_ = 5e-4;
//    auto optim = std::make_shared<torch::optim::Adam>(reprModel->parameters(), opt);
    auto optim = std::make_shared<torch::optim::SGD>(model->parameters(), opt);
    for (auto& entry : model->named_parameters()) {
        std::cout << "optimize " << entry.key() << std::endl;
    }
    args.torchOptim_ = optim;

    auto lr = &(optim->options.learning_rate_);
    args.lrPtrGetter_ = [=]() { return lr; };

    int batchSize = opts_[BatchSizeKey];
    auto dloaderOptions = torch::data::DataLoaderOptions(batchSize);
    args.dloaderOptions_ = std::move(dloaderOptions);

    auto optimizer = std::make_shared<experiments::DefaultOptimizer<TransT>>(args);
    int representationsIterations = opts_[NIterationsKey][1];
    int reportsPerEpoch = opts_[ReportsPerEpochKey];
    attachReprListeners(optimizer, 50000 / batchSize / reportsPerEpoch, representationsIterations, lr_, lr_);
//    attachDefaultListeners(optimizer, 50000 / batchSize / 10, "lenet_em_conv_checkpoint.pt");
    return optimizer;
}


namespace {

    class CatBoostOptimizer : public experiments::Optimizer {
    public:

        explicit CatBoostOptimizer(std::string catboostOptions,
            uint64_t seed,
            double lambda,
            double dropOut,
            Monom::MonomType monomType
            )
        : catBoostOptions_(std::move(catboostOptions))
        , seed_(seed)
        , lambda_(lambda)
        , drouput_(dropOut)
        , monomType_(monomType) {

        }


        void train(TensorPairDataset &trainDs,
                   TensorPairDataset &validationDs,
                   LossPtr loss,
                   experiments::ModelPtr model) const override {
            auto castedModel = std::dynamic_pointer_cast<experiments::ConvModel>(model);
            auto classifier = castedModel->classifier();
            auto polynomModel = std::dynamic_pointer_cast<PolynomModel>(classifier->classifier());

            int nTrainRows = trainDs.size().value();
            int nValidationRows = validationDs.size().value();
            auto trainData = trainDs.data().reshape({nTrainRows, -1}).to(torch::kCPU).t().contiguous();
            auto validationData = validationDs.data().reshape({nValidationRows, -1}).to(torch::kCPU).t().contiguous();

            auto trainBaseline = maybeMakeBaseline(trainDs.data().reshape({nTrainRows, -1}), classifier);
            auto validationBaseline = maybeMakeBaseline(validationDs.data().reshape({nValidationRows, -1}), classifier);
            auto trainBaselineDim = (!trainBaseline.sizes().empty()) ? trainBaseline.size(0) : 0;
            auto validationBaselineDim = (!validationBaseline.sizes().empty()) ? validationBaseline.size(0) : 0;

            auto trainTargets = trainDs.targets().to(torch::kCPU, torch::kFloat32).contiguous();
            auto validationTargets = validationDs.targets().to(torch::kCPU, torch::kFloat32).contiguous();

            const int64_t featuresCount = trainData.size(0);

            // TODO
//            std::vector<int> usedFeatures;
//            usedFeatures.resize(featuresCount);
//            std::iota(usedFeatures.begin(), usedFeatures.end(), 0);
//
//            if (drouput_ > 0) {
//                std::default_random_engine engine_(seed_);
//                std::shuffle(usedFeatures.begin(), usedFeatures.end(), engine_);
//                usedFeatures.resize(usedFeatures.size() * (1.0 - drouput_));
//            }

            TDataSet trainPool = MakePool(
                    featuresCount,
                    nTrainRows,
                    trainData.data<float>(),
                    trainTargets.data<float>(),
                    nullptr,
                    trainBaselineDim != 0 ? trainBaseline.data<float>() : nullptr,
                    trainBaselineDim);
            TDataSet validationPool = MakePool(
                    featuresCount,
                    nValidationRows,
                    validationData.data<float>(),
                    validationTargets.data<float>(),
                    nullptr,
                    validationBaselineDim != 0 ? validationBaseline.data<float>() : nullptr,
                    validationBaselineDim);

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
            for (int k = 0; k < fCount; ++k) {
                std::cout << featureIds[k] / total << " ";
            }
            std::cout << std::endl << "===============" << std::endl;
            std::cout << std::endl << "Polynom values hist" << std::endl;
            polynom->PrintHistogram();
            std::cout << std::endl << "===============" << std::endl;


            polynomModel->reset(polynom);
        }

        void train(TensorPairDataset &trainDs,
                   LossPtr loss,
                   experiments::ModelPtr model) const override {
            // TODO
            throw std::runtime_error("TODO. Should be previous implementation");
        }

        // XXX is it even used?
//        void train(const TensorPairDataset& ds,
//                   experiments::ModelPtr model) const {
//            auto classifier = dynamic_cast<experiments::Classifier*>(model.get());
//            auto polynomModel = dynamic_cast<PolynomModel*>(classifier->classifier().get());
//
//            const int samplesCount = ds.data().size(0);
//            const int validationSamplesCount = validationDs_.data().size(0);
//            auto yDim = TorchHelpers::totalSize(ds.data()) / samplesCount;
//            auto learnData = ds.data().reshape({samplesCount, yDim}).to(torch::kCPU).contiguous();
//            auto testData = testDs.data().reshape({testsamplesCount, yDim}).to(torch::kCPU).contiguous();
//            auto labels = Vec(ds.targets().to(torch::kCPU, torch::kFloat32).contiguous());
//            auto testLabels = Vec(testDs.targets().to(torch::kCPU, torch::kFloat32).contiguous());
//
//            auto learnBaseline = maybeMakeBaseline(learnData, classifier);
//            auto testBaseline = maybeMakeBaseline(testData, classifier);
//            std::vector<int> learnIndices(samplesCount);
//            std::vector<int> testIndices(testsamplesCount);
//            std::iota(learnIndices.begin(), learnIndices.end(), 0);
//            std::iota(testIndices.begin(), testIndices.end(), 0);
//
//
//            auto labelsRef = labels.arrayRef();
//            auto testLabelsRef = testLabels.arrayRef();
//
//            const int64_t featuresCount = yDim;
//
//            std::vector<float> learn(learnIndices.size() * featuresCount);
//            std::vector<float> test(testIndices.size() * featuresCount);
//
//            std::vector<int> usedFeatures;
//            usedFeatures.resize(featuresCount);
//            std::iota(usedFeatures.begin(), usedFeatures.end(), 0);
//            gather(learnData, learnIndices, featuresCount, learn, usedFeatures);
//            gather(testData, testIndices, featuresCount, test, usedFeatures);
//
//
//            auto baselineDim = learnBaseline.size() / labelsRef.size();
//            TDataSet trainPool = MakePool(learn, labelsRef, nullptr,
//                learnBaseline.size() ? learnBaseline.data() : nullptr,
//                baselineDim);
//            TDataSet testPool = MakePool(test, testLabelsRef, nullptr,
//                testBaseline.size() ? testBaseline.data() : nullptr,
//                baselineDim);
//
//            auto catboost = Train(trainPool, testPool, catBoostOptions_);
//            auto polynom = std::make_shared<Polynom>(monomType_, PolynomBuilder().AddEnsemble(catboost).Build());
//            polynom->Lambda_ = lambda_;
//            std::cout << "Model size: " << catboost.Trees.size() << std::endl;
//            std::cout << "Polynom size: " << polynom->Ensemble_.size() << std::endl;
//            polynomModel->reset(polynom);
//        }

    private:
//        inline void gather(torch::Tensor data, VecRef<int> indices, int64_t featuresCount, VecRef<float> dst, const std::vector<int>& activeFeatures) const {
//            for (uint64_t sample = 0; sample < indices.size(); ++sample) {
//                Vec features = Vec(data[indices[sample]]);
//                VERIFY(features.size() == featuresCount, "err");
//                auto featuresRef = features.arrayRef();
//                for (auto f : activeFeatures) {
//                    dst[f * indices.size() + sample] = featuresRef[f];
//                }
//            }
//        }
//
//        std::vector<float> gatherBaseline(ConstVecRef<float> baseline, ConstVecRef<int> indices, int dim) const {
//            std::vector<float> result;
//            if (!baseline.empty()) {
//                const int samplCount = baseline.size() / dim;
//                result.resize(indices.size() * dim);
//                for (int currentDim = 0; currentDim < dim; ++currentDim) {
//                    for (size_t i = 0; i < indices.size(); ++i) {
//                        result[currentDim * indices.size() + i] = baseline[currentDim * samplCount  + indices[i]];
//                    }
//                }
//            }
//            return result;
//        }

        torch::Tensor maybeMakeBaseline(torch::Tensor data, const experiments::ClassifierPtr& classifier) const {
            if (classifier->baseline()) {
                auto baseline = classifier->baseline();
                data = experiments::correctDevice(data, baseline);
                auto baselineTensor = classifier->baseline()->forward(data).to(torch::kCPU).t().contiguous();
                return baselineTensor;
            }
            return torch::zeros({});
        }

    private:
        std::string catBoostOptions_;
        uint64_t seed_ = 0;
        double lambda_ = 1.0;
        double drouput_ = 0.0;
        Monom::MonomType monomType_;
    };
}

experiments::OptimizerPtr CatBoostNN::getDecisionOptimizer(const experiments::ModelPtr& decisionModel) {
    seed_ += 10000;
    std::string params;
    if (Init_) {
        params = opts_[CatboostParamsKey][InitParamsKey].dump();
        Init_ = false;
    } else {
        params = opts_[CatboostParamsKey][IntermediateParamsKey].dump();
    }

    return std::make_shared<CatBoostOptimizer>(
        params,
        seed_,
        ((double)opts_[ModelKey][ClassifierKey][ClassifierMainKey][LambdaKey]) * lambdaMult_,
        opts_[DropoutKey],
        Monom::getMonomType(opts_[ModelKey][ClassifierKey][ClassifierMainKey][MonomTypeKey])
        );
}

void CatBoostNN::train(
        TensorPairDataset &trainDs,
        const LossPtr& loss) {
    lr_ = opts_[SgdStepSizeKey];

    initialTrainRepr(trainDs, loss);

    trainDecision(trainDs, loss);

    for (uint32_t i = 0; i < iterations_; ++i) {
        std::cout << "EM iteration: " << i << std::endl;

        fireScheduledParamModifiers(i);

        lr_ = opts_[SgdStepSizeKey];

        trainRepr(trainDs, loss);

        std::cout << "Repr was trained, calling listeners " << i << std::endl;
//        model_->classifier()->dumpClassifierScale();

        fireListeners(2 * i);
        std::cout << "========== " << i << std::endl;

        trainDecision(trainDs, loss);

        std::cout << "Decision was trained " << i << std::endl;

        fireListeners(2 * i + 1);
        std::cout << "========== " << i << std::endl;

    }
}

TensorPairDataset CatBoostNN::getRepr(TensorPairDataset &ds, const experiments::ModelPtr &reprModel) {
    auto mds = ds.map(reprTransform_);
    auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(256));
    std::vector<torch::Tensor> reprList;
    std::vector<torch::Tensor> targetsList;

    for (auto& batch : *dloader) {
        auto res = reprModel->forward(batch.data);
        auto target = batch.target;
        reprList.push_back(res);
        targetsList.push_back(target);
    }

    auto repr = torch::cat(reprList, 0);
    auto targets = torch::cat(targetsList, 0);
    reprList.clear();
    targetsList.clear();

    return {repr, targets};
}

void CatBoostNN::trainDecision(TensorPairDataset& ds, const LossPtr& loss) {
    auto representationsModel = model_->conv();
    auto decisionModel = model_->classifier();
    representationsModel->train(false);



    if (model_->classifier()->baseline()) {
        model_->classifier()->enableBaselineTrain(false);
    }
    model_->classifier()->enableScaleTrain(false);
    model_->classifier()->classifier()->train(true);

    std::cout << "    getting representations" << std::endl;

    auto trainDsRepr = getRepr(ds, representationsModel);
    auto validationDsRepr = getRepr(validationDs_, representationsModel);

    std::cout << "    optimizing decision model" << std::endl;

    auto decisionFuncOptimizer = getDecisionOptimizer(decisionModel);
    decisionFuncOptimizer->train(trainDsRepr, validationDsRepr, loss, model_);

}

void CatBoostNN::trainRepr(TensorPairDataset& ds, const LossPtr& loss) {
    auto representationsModel = model_->conv();
    at::Tensor prevX;
    {
        std::vector<at::Tensor> kolbosa;
        for (const at::Tensor& t: representationsModel->parameters(true)) {
            if (!t.dim())
                continue;
            kolbosa.push_back(t.flatten());
        }
        prevX = torch::cat(kolbosa, 0).clone();
    }
    representationsModel->train(true);
    auto decisionModel = model_->classifier();
    decisionModel->classifier()->train(false);

    if (decisionModel->baseline()) {
        decisionModel->enableBaselineTrain(true);
    }
    decisionModel->enableScaleTrain(trainScale_);

    std::cout << "    optimizing representation model" << std::endl;

    LossPtr representationLoss = makeRepresentationLoss(decisionModel, loss);
    auto representationOptimizer = getReprOptimizer(model_);

    model_->classifier()->printScale();
    representationOptimizer->train(ds, representationLoss, representationsModel);
    model_->classifier()->printScale();
    {
        std::vector<at::Tensor> kolbosa;
        for (const at::Tensor& t: representationsModel->parameters(true)) {
            if (!t.dim())
                continue;
            kolbosa.push_back(t.flatten());
        }
        at::Tensor currentX = torch::cat(kolbosa, 0).clone();
        std::cout << "Optimization step length: " << (currentX - prevX).norm(2) << " total CNN parameters count: " << currentX.sizes() << std::endl;
    };
}
experiments::ModelPtr CatBoostNN::trainFinalDecision(TensorPairDataset& learn, const TensorPairDataset& test) {
    throw std::runtime_error("TODO");
//    auto optimizer = std::make_shared<CatBoostOptimizer>(
//        opts_[CatboostParamsKey][FinalParamsKey].dump(),
//        seed_,
//        1e10,
//        0.0,
//        Monom::getMonomType(opts_[ModelKey][ClassifierKey][ClassifierMainKey][MonomTypeKey])
//    );
//    experiments::ClassifierPtr classifier;
//    auto baseline = model_->classifier()->baseline();
//    if (baseline) {
//        classifier = experiments::makeClassifierWithBaseline<PolynomModel>(model_->classifier()->baseline(), std::make_shared<Polynom>());
//    } else {
//        classifier = experiments::makeClassifier<PolynomModel>(std::make_shared<Polynom>());
//    }
//    optimizer->train(learn, std::make_shared<ZeroLoss>(), classifier);
//    return classifier;
}
void CatBoostNN::setLambda(double lambda) {
    auto model = dynamic_cast<PolynomModel*>(model_->classifier()->classifier().get());
    VERIFY(model != nullptr, "model is not polynom");
    model->setLambda(lambda * lambdaMult_);

}

using namespace experiments;

void CatBoostNN::initialTrainRepr(TensorPairDataset& ds, const LossPtr& loss) {
    iter_ = 3;

    if (initClassifier_) {
        auto model = std::make_shared<ConvModel>(model_->conv(), initClassifier_);
        model->train(true);
        LossPtr representationLoss = makeRepresentationLoss(initClassifier_, loss);
        auto representationOptimizer = getReprOptimizer(model_->conv());
        representationOptimizer->train(ds, representationLoss, model_->conv());
    }

    if (model_->classifier()->baseline()) {
        iter_ = 1;
        trainRepr(ds, loss);
    }

    iter_ =  opts_[NIterationsKey][1];
}

void CatBoostNN::fireScheduledParamModifiers(int iter) {
    const std::vector<json> paramModifiers = opts_[ScheduledParamModifiersKey];

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
                    setField(opts_, field, values[i]);
                    break;
                }
            }
        } else {
            std::vector<int> values = modifier[ValuesKey];
            for (int i = 0; i < iters.size(); ++i) {
                if (iters[i] == iter) {
                    std::cout << "changing " << field << " on iter " << iter << " to " << values[i] << std::endl;
                    setField(opts_, field, values[i]);
                    break;
                }
            }
        }
    }
}

//
//std::pair<torch::Tensor, torch::Tensor> CatBoostNN::representation(TensorPairDataset& ds) {
//    auto dloader = torch::data::make_data_loader(ds, torch::data::DataLoaderOptions(100));
//    auto device = representationsModel_->parameters().data()->device();
//    std::vector<torch::Tensor> reprList;
//    std::vector<torch::Tensor> targetsList;
//
//    for (auto& batch : *dloader) {
//        auto res = representationsModel_->forward(batch.data.to(device));
//        auto target = batch.target.to(device);
//        reprList.push_back(res);
//        targetsList.push_back(target);
//    }
//
//    auto repr = torch::cat(reprList, 0);
//    auto targets = torch::cat(targetsList, 0);
//}
