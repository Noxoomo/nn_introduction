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
#include <algorithm>

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

//  auto lrDecayListener = std::make_shared<LrLinearDecayOptimizerListener>(startLr, endLr, epochs);
//  optimizer->registerListener(lrDecayListener);
}

static torch::Tensor maybeMakeBaseline(torch::Tensor data, const experiments::ClassifierPtr& classifier,
        bool transpose = false) {
    if (classifier->baseline()) {
        auto baseline = classifier->baseline();
        data = experiments::correctDevice(data, baseline).view({data.size(0), -1});
        auto baselineTensor = classifier->baseline()->forward(data);
        if (transpose) {
            baselineTensor = baselineTensor.t();
        }
        baselineTensor = baselineTensor.to(torch::kCPU).contiguous();
        return baselineTensor;
    }
    return torch::zeros({});
}

static TensorPairDataset binarizeDs(TensorPairDataset &ds, double border) {
    auto newData = torch::gt(ds.data(), border).to(torch::kFloat32).view({ds.data().size(0), -1});
    return TensorPairDataset(newData, ds.targets());
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

experiments::OptimizerPtr CatBoostNN::getBaselineOptimizer(const experiments::ModelPtr &model) {
    // test transform just performs stacking
    auto transform = getDefaultCifar10TestTransform();
    using TransT = decltype(transform);

    experiments::OptimizerArgs<TransT> args(transform, iter_);
//
    double lr = 0.001;
    torch::optim::SGDOptions opt(lr);
    opt.momentum_ = 0.9;
//    torch::optim::AdamOptions opt(lr);
    opt.weight_decay_ = 5e-4;
//    auto optim = std::make_shared<torch::optim::Adam>(reprModel->parameters(), opt);
    auto optim = std::make_shared<torch::optim::SGD>(model->parameters(), opt);
    args.torchOptim_ = optim;

    {
        auto lr = &(optim->options.learning_rate_);
        args.lrPtrGetter_ = [=]() { return lr; };
    }

    int batchSize = opts_[BatchSizeKey];
    auto dloaderOptions = torch::data::DataLoaderOptions(batchSize);
    args.dloaderOptions_ = std::move(dloaderOptions);

    auto optimizer = std::make_shared<experiments::DefaultOptimizer<TransT>>(args);
    int representationsIterations = 5;
    int reportsPerEpoch = opts_[ReportsPerEpochKey];
    attachReprListeners(optimizer, 50000 / batchSize / reportsPerEpoch, representationsIterations, lr, lr);
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
            auto castedModel = std::dynamic_pointer_cast<experiments::EmModel>(model);
            auto classifier = castedModel->classifier();
            auto polynomModel = std::dynamic_pointer_cast<PolynomModel>(classifier->classifier());

            int nTrainRows = trainDs.size().value();
            int nValidationRows = validationDs.size().value();
            auto trainData = trainDs.data().reshape({nTrainRows, -1}).to(torch::kCPU).t().contiguous();
            auto validationData = validationDs.data().reshape({nValidationRows, -1}).to(torch::kCPU).t().contiguous();

            auto trainBaselineData = binarizeDs(trainDs, 0).data().reshape({nTrainRows, -1});
            auto validationBaselineData = binarizeDs(validationDs, 0).data().reshape({nValidationRows, -1});
            auto trainBaseline = maybeMakeBaseline(trainBaselineData, classifier, true);
            auto validationBaseline = maybeMakeBaseline(validationBaselineData, classifier, true);
            auto trainBaselineDim = (!trainBaseline.sizes().empty()) ? trainBaseline.size(0) : 0;
            auto validationBaselineDim = (!validationBaseline.sizes().empty()) ? validationBaseline.size(0) : 0;

            auto trainTargets = trainDs.targets().to(torch::kCPU, torch::kFloat32).contiguous();
            auto validationTargets = validationDs.targets().to(torch::kCPU, torch::kFloat32).contiguous();

            const int64_t featuresCount = trainData.size(0);

            std::cout << "train data shape: " << trainData.sizes() << std::endl;
            std::cout << "validation data shape: " << validationData.sizes() << std::endl;
            std::cout << "train baseline shape: " << trainBaseline.sizes() << std::endl;
            std::cout << "validation baseline shape: " << validationBaseline.sizes() << std::endl;

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
                    trainData.data_ptr<float>(),
                    trainTargets.data_ptr<float>(),
                    nullptr,
                    trainBaselineDim != 0 ? trainBaseline.data_ptr<float>() : nullptr,
                    trainBaselineDim);
            TDataSet validationPool = MakePool(
                    featuresCount,
                    nValidationRows,
                    validationData.data_ptr<float>(),
                    validationTargets.data_ptr<float>(),
                    nullptr,
                    validationBaselineDim != 0 ? validationBaseline.data_ptr<float>() : nullptr,
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
            for (int k = 0; k <= fCount; ++k) {
                std::cout << featureIds[k] / total << " ";
            }
            std::cout << std::endl << "===============" << std::endl;
            std::cout << std::endl << "Polynom values hist" << std::endl;
            polynom->PrintHistogram();
            std::cout << std::endl << "===============" << std::endl;

//            exit(1);
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

static void setLastBias(experiments::ModelPtr model_, TensorPairDataset &ds) {
    auto data = ds.data().view({ds.data().size(0), -1}).t().to(torch::kCPU).contiguous();
    auto dataAccessor = data.accessor<float, 2>();

    torch::Tensor biases = torch::zeros({data.size(0)}, torch::kFloat32);
    auto biasesAccessor = biases.accessor<float, 1>();

    for (int f = 0; f < data.size(0); f++) {
        std::vector<float> values;
        for (int i = 0; i < data.size(1); i++) {
            values.push_back(dataAccessor[f][i]);
        }
        std::sort(values.begin(), values.end());
        int n = values.size();
        float bias = 0;
        if (n % 2 == 1) {
            bias = values[n / 2];
        } else {
            bias = (values[n / 2 - 1] + values[n / 2]) / 2;
        }
        biasesAccessor[f] = -bias;
    }

    std::cout << "bias computed" << std::endl;

    model_->setLastBias(biases);
}

static void reprPrintStats(TensorPairDataset &ds) {
    auto data = ds.data().view({ds.data().size(0), -1}).to(torch::kCPU).contiguous();

    auto dataAccessor = data.accessor<float, 2>();

    struct counts {
        int neg;
        int zero;
        int pos;
    };

    std::cout << "repr stats:" << std::endl;
    for (int f = 0; f < data.size(1); f++) {
        counts cnt = {0, 0, 0};
        for (int i = 0; i < data.size(0); i++) {
            if (dataAccessor[i][f] < 0) {
                cnt.neg++;
            } else if (dataAccessor[i][f] < 1e-9) {
                cnt.zero++;
            } else {
                cnt.pos++;
            }
        }
        std::cout << f << ": (< 0) " << cnt.neg << ", (= 0) " << cnt.zero << ", (> 0)" << cnt.pos << std::endl;
    }
}

void CatBoostNN::trainDecision(TensorPairDataset& ds, const LossPtr& loss) {
    auto representationsModel = model_->conv();
    auto decisionModel = model_->classifier();
    representationsModel->train(false);

    static bool needBias = false;

    if (model_->classifier()->baseline()) {
        model_->classifier()->enableBaselineTrain(false);
    }
    model_->classifier()->enableScaleTrain(trainScale_);
    model_->classifier()->classifier()->train(true);

    std::cout << "    getting representations" << std::endl;

//    representationsModel->lastNonlinearity(false);
    auto trainDsRepr = getRepr(ds, representationsModel);
    if (needBias) {
        std::cout << "\n\n Setting Bias \n\n" << std::endl;
        reprPrintStats(trainDsRepr);
        setLastBias(representationsModel, trainDsRepr);
        trainDsRepr = getRepr(ds, representationsModel);
        needBias = false;
        reprPrintStats(trainDsRepr);
    }
    auto validationDsRepr = getRepr(validationDs_, representationsModel);

    std::cout << "    optimizing decision model" << std::endl;

    trainBaseline(trainDsRepr, loss);
    auto decisionFuncOptimizer = getDecisionOptimizer(decisionModel);
    decisionFuncOptimizer->train(trainDsRepr, validationDsRepr, loss, model_);
}

void CatBoostNN::setScale(TensorPairDataset &ds) {
    auto representationsModel = model_->conv();
    auto decisionModel = model_->classifier();
//    auto classifier = decisionModel->classifier();

    representationsModel->train(false);
    auto reprDs = getRepr(ds, representationsModel);
    auto reprData = reprDs.data();
    auto reprTarget = reprDs.targets().to(torch::kCPU).contiguous();

//    auto baseline = maybeMakeBaseline(reprData, decisionModel).view({reprData.size(0), -1}).to(torch::kCPU).contiguous();

    std::vector<torch::Tensor> stack;
    for (int64_t offset = 0; offset < reprTarget.size(0); offset += 1024) {
        stack.push_back(decisionModel->forward(reprData.slice(0, offset, std::min(offset + 1024, reprData.size(0)))).to(torch::kCPU).contiguous());
    }
    auto h_x = torch::cat(stack, 0);

    const at::TensorAccessor<float, 2> &h_x_accessor = h_x.accessor<float, 2>();
    const at::TensorAccessor<int64_t, 1> &target_accessor = reprTarget.accessor<int64_t, 1>();

    std::cout << "train data shape: " << h_x.sizes() << std::endl;
//    std::cout << "train baseline shape: " << baseline.sizes() << std::endl;

    double scale = 1;
    {
        double p[h_x_accessor.size(1)];
        for (int it = 0; it < 10; it++) {
            double dJ_dalpha = 0;
            double dJ_dalpha2 = 0;
            for (int64_t i = 0; i < h_x_accessor.size(0); i++) {
                double P = 0;
                for (int j = 0; j < (int)h_x_accessor.size(1); j++) {
                    P += exp(scale * h_x_accessor[i][j]);
                }
                for (int j = 0; j < (int)h_x_accessor.size(1); j++) {
                    p[j] = exp(scale * h_x_accessor[i][j]) / P;
                }
                int yi = target_accessor[i];
                for (int j = 0; j < (int)h_x_accessor.size(1); j++) {
                    dJ_dalpha += (h_x_accessor[i][yi] - h_x_accessor[i][j]) * p[j];
                }
                double dJ_dalpha2_1 = 0;
                double dJ_dalpha2_2 = 0;
                double dJ_dalpha2_3 = 0;
                for (int j = 0; j < (int)h_x_accessor.size(1); j++) {
                    dJ_dalpha2_1 += (h_x_accessor[i][yi] - h_x_accessor[i][j]) * h_x_accessor[i][j] * p[j];
                    dJ_dalpha2_2 += h_x_accessor[i][j] * p[j];
                    dJ_dalpha2_3 += (h_x_accessor[i][yi] - h_x_accessor[i][j]) * p[j];
                }
                dJ_dalpha2 += dJ_dalpha2_1 - dJ_dalpha2_2 * dJ_dalpha2_3;
            }
            std::cout << dJ_dalpha << " " << dJ_dalpha2 << " ";
            scale -= dJ_dalpha / dJ_dalpha2;
            std::cout << scale << std::endl;
        }
    }
    {
        double originalScore = 0;
        double scaledScore = 0;
        double p_origin[h_x_accessor.size(1)];
        double p_scaled[h_x_accessor.size(1)];
        for (int64_t i = 0; i < h_x_accessor.size(0); i++) {
            int yi = target_accessor[i];
            double P_origin = 0;
            double P_scaled = 0;
            for (int j = 0; j < (int)h_x_accessor.size(1); j++) {
                P_scaled += exp(scale * h_x_accessor[i][j]);
                P_origin += exp(h_x_accessor[i][j]);
            }
            for (int j = 0; j < (int)h_x_accessor.size(1); j++) {
                p_scaled[j] = exp(scale * h_x_accessor[i][j]) / P_scaled;
                p_origin[j] = exp(h_x_accessor[i][j]) / P_origin;
            }
            originalScore += log(p_origin[yi]);
            scaledScore += log(p_scaled[yi]);
        }
        std::cout << "Scaled score: " << scaledScore << ", original score: " << originalScore
                  << ", scale: " << scale << std::endl;
    }

    decisionModel->setScale(scale);
    decisionModel->printScale();
}

void CatBoostNN::trainBaseline(TensorPairDataset &ds, const LossPtr &loss) {
    auto decisionModel = model_->classifier();
    auto baselineModel = decisionModel->baseline();

    if (!baselineModel) {
        return;
    }

    model_->conv()->train(false);
    decisionModel->classifier()->train(false);
    baselineModel->train(true);

    std::cout << "binarizing ds" << std::endl;
    auto binarizedDs = binarizeDs(ds, 0);
    std::cout << "binarized ds shape: " << binarizedDs.data().sizes() << std::endl;

    // TODO baseline optimizer, but it uses the same for now
    std::cout << "    training baseline" << std::endl;
    auto optim = getBaselineOptimizer(baselineModel);
    optim->train(binarizedDs, loss, baselineModel);
    std::cout << "    done training baseline" << std::endl;
}

void CatBoostNN::trainRepr(TensorPairDataset& ds, const LossPtr& loss) {
    auto representationsModel = model_->conv();
    auto decisionModel = model_->classifier();
    decisionModel->train(false);
//    if (decisionModel->baseline()) {
//        decisionModel->enableBaselineTrain(true);
//    }

    decisionModel->setScale(1);
    setScale(ds);

//    if (!decisionModel->baseline() || trainedBaseline_) {
//        decisionModel->setScale(1);
//        setScale(ds);
//    } else {
//        trainedBaseline_ = true;
//    }
//    decisionModel->enableScaleTrain(true);

    std::cout << "    optimizing representation model" << std::endl;
//    decisionModel->printScale();

    representationsModel->train(true);
    LossPtr representationLoss = makeRepresentationLoss(decisionModel, loss);
    auto representationOptimizer = getReprOptimizer(model_);

    representationOptimizer->train(ds, representationLoss, representationsModel);
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
        auto model = std::make_shared<EmModel>(model_->conv(), initClassifier_);
        model->train(true);
        LossPtr representationLoss = makeRepresentationLoss(initClassifier_, loss);
        auto representationOptimizer = getReprOptimizer(model_->conv());
        representationOptimizer->train(ds, representationLoss, model_->conv());
    }

//    if (model_->classifier()->baseline()) {
//        iter_ = 1;
//        trainRepr(ds, loss);
//    }

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
