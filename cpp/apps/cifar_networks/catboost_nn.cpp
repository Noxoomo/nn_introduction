#include "catboost_nn.h"

#include <utility>

#include <random>
#include <core/vec.h>
#include <catboost_wrapper.h>
#include <models/polynom/polynom.h>
#include <cifar_nn/polynom_model.h>

experiments::ModelPtr CatBoostNN::getTrainedModel(TensorPairDataset& ds, const LossPtr& loss) {
    train(ds, loss);
    return model_;
}

inline TPool MakePool(const std::vector<float>& features,
                      const std::vector<float>& labels,
                      float* weights = nullptr) {
    const int fCount = features.size() / labels.size();
    TPool pool;
    pool.Features = features.data();
    pool.Labels =  labels.data();
    pool.FeaturesCount = fCount;
    pool.SamplesCount = labels.size();
    pool.Weights = weights;
    return pool;
}

experiments::OptimizerPtr CatBoostNN::getReprOptimizer(const experiments::ModelPtr& reprModel) {
    auto transform = getDefaultCifar10TrainTransform();
    using TransT = decltype(transform);

    experiments::OptimizerArgs<TransT> args(transform, opts_.representationsIterations, device_);

    torch::optim::AdamOptions opt(opts_.adamStep);
//        opt.weight_decay_ = 5e-4;
    auto optim = std::make_shared<torch::optim::Adam>(reprModel->parameters(), opt);
    args.torchOptim_ = optim;

    auto lr = &(optim->options.learning_rate_);
    args.lrPtrGetter_ = [=]() { return lr; };

    const auto batchSize= opts_.batchSize;
    auto dloaderOptions = torch::data::DataLoaderOptions(batchSize);
    args.dloaderOptions_ = std::move(dloaderOptions);

    auto optimizer = std::make_shared<experiments::DefaultOptimizer<TransT>>(args);
    attachDefaultListeners(optimizer, 50000 / batchSize / 10, "lenet_em_conv_checkpoint.pt");
    return optimizer;
}


namespace {

    class CatBoostOptimizer : public experiments::Optimizer {
    public:

        explicit CatBoostOptimizer(std::string catboostOptions,
            uint64_t seed,
            double lambda)
        : catBoostOptions_(std::move(catboostOptions))
        , seed_(seed)
        , lambda_(lambda) {

        }


        void train(TensorPairDataset& ds,
                   LossPtr loss,
                   experiments::ModelPtr model) const override {
            auto polynomModel = dynamic_cast<PolynomModel*>(model.get());
            auto data = ds.data();
            const int samplesCount = data.size(0);
            auto yDim = TorchHelpers::totalSize(data) / samplesCount;
            data = data.reshape({samplesCount, yDim}).to(torch::kCPU);
            auto labels = Vec(ds.targets().to(torch::kCPU, torch::kFloat32));

            std::default_random_engine engine_(seed_);
            auto poisson_ = std::poisson_distribution<int>(1);


            std::vector<int> learnIndices;
            std::vector<int> testIndices;

            std::vector<float> learnTargets;
            std::vector<float> testTargets;
            std::vector<float> learnWeights;

            auto labelsRef = labels.arrayRef();

            for (int32_t i = 0; i < labels.size(); ++i) {
                const int w =  poisson_(engine_);
                const float t = labelsRef[i];
                if (w > 0) {
                    learnIndices.push_back(i);
                    learnTargets.push_back(t);
                } else {
                    testIndices.push_back(i);
                    testTargets.push_back(t);
                }
            }

            const int64_t featuresCount = data.size(1);

            std::vector<float> learn(learnIndices.size() * featuresCount);
            std::vector<float> test(testIndices.size() * featuresCount);

            gather(data, learnIndices, featuresCount, learn);
            gather(data, testIndices, featuresCount, test);


            TPool trainPool = MakePool(learn, learnTargets, learnWeights.data());
            TPool testPool = MakePool(test, testTargets);

            auto catboost = Train(trainPool, testPool, catBoostOptions_);
            Polynom polynom(PolynomBuilder().AddEnsemble(catboost).Build());
            polynom.Lambda_ = lambda_;
            std::cout << "Model size: " << catboost.Trees.size() << std::endl;
            std::cout << "Polynom size: " << polynom.Ensemble_.size() << std::endl;
            polynomModel->polynom_ = std::make_shared<Polynom>(polynom);
        }

        void registerListener(std::shared_ptr<experiments::OptimizerBatchListener>) override {
            //pass, no batchs
        }
        void registerListener(std::shared_ptr<experiments::OptimizerEpochListener>) override {
            //pass, no epochs
        }
    private:
        inline void gather(torch::Tensor data, VecRef<int> indices, int64_t featuresCount, VecRef<float> dst) const {
            for (uint64_t sample = 0; sample < indices.size(); ++sample) {
                Vec features = Vec(data[indices[sample]]);
                VERIFY(features.size() == featuresCount, "err");
                auto featuresRef = features.arrayRef();
                for (uint64_t f = 0; f < featuresCount; ++f) {
                    dst[f * indices.size() + sample] = featuresRef[f];
                }
            }
        }

    private:
        std::string catBoostOptions_;
        uint64_t seed_ = 0;
        double lambda_ = 1.0;
    };
}


inline std::string readFile(const std::string& path) {
    std::ifstream in(path);
    std::stringstream strStream;
    strStream << in.rdbuf(); //read the file
    std::string params = strStream.str();
    return params;
}

experiments::OptimizerPtr CatBoostNN::getDecisionOptimizer(const experiments::ModelPtr& decisionModel) {
    seed_ += 10000;
    return std::make_shared<CatBoostOptimizer>(
        readFile(opts_.catboostParamsFile),
        seed_,
        opts_.lambda_
        );
}

