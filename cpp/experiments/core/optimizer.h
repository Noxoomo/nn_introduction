#pragma once

#include "tensor_pair_dataset.h"
#include "loss.h"
#include "model.h"

#include <torch/torch.h>

#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <algorithm>

namespace experiments {

// OptimizerListener

class OptimizerBatchListener {
public:
    OptimizerBatchListener() = default;

    virtual void batchReset() = 0;

    virtual void onBatch(int epoch, int batchId, float batchLoss) = 0;

    virtual ~OptimizerBatchListener() = default;
};

class OptimizerEpochListener {
public:
    OptimizerEpochListener() = default;

    virtual void epochReset() = 0;

    virtual void onEpoch(int epoch, double* lr, ModelPtr model) = 0;

    virtual ~OptimizerEpochListener() = default;
};

class BatchReportOptimizerListener : public OptimizerBatchListener {
public:
    explicit BatchReportOptimizerListener(int nBatchesReport);

    void batchReset() override;

    void onBatch(int epoch, int batchId, float batchLoss) override;

private:
    int nBatchesReport_;
    float runningLoss_;

};

class EpochReportOptimizerListener : public OptimizerEpochListener {
public:
    EpochReportOptimizerListener();

    void epochReset() override;

    void onEpoch(int epoch, double* lr, ModelPtr model) override;

private:
    std::chrono::high_resolution_clock::time_point epochStartTime_;
};

class LrDecayOptimizerListener : public OptimizerEpochListener {
public:
    LrDecayOptimizerListener(
            std::vector<double> newLr,
            std::vector<int> decayEpochs);

    void epochReset() override;

    void onEpoch(int epoch, double* lr, ModelPtr model) override;

private:
    std::vector<double> newLr_;
    std::vector<int> decayEpochs_;
};

class ModelSaveOptimizerListener : public OptimizerEpochListener {
public:
    ModelSaveOptimizerListener(int nEpochsSave,
            std::string path);

    void epochReset() override;

    void onEpoch(int epoch, double* lr, ModelPtr model) override;

private:
    int nEpochsSave_;
    std::string path_;
};

class EpochEndCallback : public experiments::OptimizerEpochListener {
public:
    template <class Callback>
    EpochEndCallback(Callback callback)
            : callback_(std::move(callback)) {

    }


    void epochReset() override {

    }

    void onEpoch(int epoch, double* lr, experiments::ModelPtr model) override {
        callback_(epoch, *model);
    }

private:
    std::function<void(int, experiments::Model& model)> callback_;
};

// Optimizer

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void train(TensorPairDataset& ds, LossPtr loss, ModelPtr model) const = 0;

    virtual void train(TensorPairDataset &trainDs, TensorPairDataset &validationDs, LossPtr loss, ModelPtr model) const {
        throw std::runtime_error("Not implemented");
    }

    void train(torch::Tensor& x, torch::Tensor& y, LossPtr loss, ModelPtr model) const {
        TensorPairDataset ds(x, y);
        this->train(ds, std::move(loss), std::move(model));
    }

    void registerListener(const std::shared_ptr<OptimizerBatchListener>& listener) {
        batchListeners_.push_back(listener);
    }

    void registerListener(const std::shared_ptr<OptimizerEpochListener>& listener) {
        epochListeners_.push_back(listener);
    }

    template <class Listener, class ... Args>
    static void emplaceEpochListener(Optimizer* optimizer, Args... args) {
        std::shared_ptr<OptimizerEpochListener> listener(new Listener(std::forward<Args>(args)...));
        optimizer->registerListener(listener);
    }


protected:
    void fireEpochResetListeners() const {
        for (auto& listener : epochListeners_) {
            listener->epochReset();
        }
    }

    void fireOnEpochListeners(int epoch, double* lr, const ModelPtr& model) const {
        std::cout << std::endl;
        for (auto& listener : epochListeners_) {
            listener->onEpoch(epoch, lr, model);
        }
        std::cout << std::endl;
    }

    void fireBatchResetListeners() const {
        for (auto& listener : batchListeners_) {
            listener->batchReset();
        }
    }

    void fireOnBatchListeners(int epoch, int batchId, float batchLoss) const {
        for (auto& listener : batchListeners_) {
            listener->onBatch(epoch, batchId, batchLoss);
        }
    }

private:
    std::vector<std::shared_ptr<OptimizerBatchListener>> batchListeners_;
    std::vector<std::shared_ptr<OptimizerEpochListener>> epochListeners_;

};

using OptimizerPtr = std::shared_ptr<Optimizer>;

// OptimizerArgs

template <typename TransformType>
struct OptimizerArgs {
    explicit OptimizerArgs(TransformType transform,
            int epochs = 10)
            : transform_(std::move(transform))
            , epochs_(epochs) {

    }

    std::shared_ptr<torch::optim::Optimizer> torchOptim_{nullptr};

    torch::data::DataLoaderOptions dloaderOptions_;
    TransformType transform_;

    std::function<double*(void)> lrPtrGetter_;

    int epochs_ = -1;
};

class NoopOptimizer : public Optimizer {
public:
    NoopOptimizer() = default;

    void train(TensorPairDataset& ds, LossPtr loss, ModelPtr model) const {
    }

    ~NoopOptimizer() = default;
};

// DefaultOptimizer

template <typename TransformType>
class DefaultOptimizer : public Optimizer {
public:
    explicit DefaultOptimizer(OptimizerArgs<TransformType> args)
            : args_(std::move(args)) {

    }

    void train(TensorPairDataset& ds, LossPtr loss, ModelPtr model) const override {
        auto mds = ds.map(args_.transform_);
        auto dloader = torch::data::make_data_loader(mds, args_.dloaderOptions_);

        for (int epoch = 0; epoch < args_.epochs_; epoch++) {
            this->fireEpochResetListeners();
            this->fireBatchResetListeners();
            int batchId = 0;
            for (const auto& batch : *dloader) {
                auto data = batch.data;
                auto target = batch.target;

                model->train(true);
                args_.torchOptim_->zero_grad();
                auto prediction = model->forward(data);
                torch::Tensor lossVal = loss->value(prediction, target);
                lossVal.backward();
                args_.torchOptim_->step();

                this->fireOnBatchListeners(epoch, batchId, lossVal.item<float>());
                batchId++;
            }
            double* lr = args_.lrPtrGetter_();
            this->fireOnEpochListeners(epoch, lr, model);
        }
    }

    ~DefaultOptimizer() override = default;

private:
    OptimizerArgs<TransformType> args_;

};

}
