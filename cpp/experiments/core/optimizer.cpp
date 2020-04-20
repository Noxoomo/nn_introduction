#include "optimizer.h"
#include "tensor_pair_dataset.h"
#include "loss.h"
#include "model.h"

#include <torch/torch.h>

#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

namespace experiments {

// BatchReportOptimizerListener

BatchReportOptimizerListener::BatchReportOptimizerListener(int nBatchesReport)
        : OptimizerBatchListener()
        , nBatchesReport_(nBatchesReport) {
    runningLoss_ = 0.;
}

void BatchReportOptimizerListener::batchReset() {
    runningLoss_ = 0.;
}

void BatchReportOptimizerListener::onBatch(int epoch, int batchId, float batchLoss) {
    runningLoss_ += batchLoss;
    if ((batchId + 1) % nBatchesReport_ != 0) {
        return;
    }
    std::cout << "[" << epoch << ", " << batchId << "] "
              << "loss: " << (runningLoss_ / nBatchesReport_) << std::endl;
    runningLoss_ = 0;
}

// EpochReportOptimizerListener

EpochReportOptimizerListener::EpochReportOptimizerListener()
        : OptimizerEpochListener() {

}

void EpochReportOptimizerListener::epochReset() {
    epochStartTime_ = std::chrono::high_resolution_clock::now();
}

void EpochReportOptimizerListener::onEpoch(int epoch, double *lr, experiments::ModelPtr model) {
    std::cout << "End of epoch #" << epoch << ", lr = " << std::setprecision(3) << (*lr);

    auto epochElapsedTime = std::chrono::high_resolution_clock::now() - epochStartTime_;
    auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(epochElapsedTime);
    std::cout << ", elapsed time since last epoch: " << elapsedTimeMs.count() << std::endl;
}

// LrDecayOptimizerListener

LrDecayOptimizerListener::LrDecayOptimizerListener(
        std::vector<double> newLr,
        std::vector<int> decayEpochs)
        : OptimizerEpochListener()
        , newLr_(std::move(newLr))
        , decayEpochs_(std::move(decayEpochs)) {
    assert(newLr_.size() == decayEpochs_.size());
}

void LrDecayOptimizerListener::epochReset() {

}

void LrDecayOptimizerListener::onEpoch(int epoch, double *lr, experiments::ModelPtr model) {
    for (int i = 0; i < decayEpochs_.size(); ++i) {
        if (epoch == decayEpochs_[i]) {
            std::cout << "Decaying lr: (" << (*lr) << " -> " << newLr_[i] << ")" << std::endl;
            *lr = newLr_[i];
            return;
        }
    }
}

// ModelSaveOptimizerListener

ModelSaveOptimizerListener::ModelSaveOptimizerListener(int nEpochsSave, std::string path)
        : OptimizerEpochListener()
        , nEpochsSave_(nEpochsSave)
        , path_(std::move(path)) {

}

void ModelSaveOptimizerListener::epochReset() {

}

void ModelSaveOptimizerListener::onEpoch(int epoch, double *lr, experiments::ModelPtr model) {
    if (epoch % nEpochsSave_ == 0) {
        std::cout << "Saving model to '" << path_ << "'" << std::endl;
        torch::save(model, path_);
    }
}

}
