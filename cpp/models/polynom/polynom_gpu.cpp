#include "polynom_gpu.h"
#include "soft_polynom.h"
#include <core/buffer.h>
#include <util/singleton.h>
#include <ostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <stdexcept>

using namespace std;
namespace {
    class PolynomForwardBackwardLog {
    public:
        PolynomForwardBackwardLog()
        : out_("features_der.txt") {

        }


        void invokeBackward(torch::Tensor batch,
                            torch::Tensor featureDer) {
            auto derNorm = featureDer.norm(2).to(torch::kCPU).data<float>()[0];
            out_ << i << " der norm " << derNorm << std::endl;
            auto batchDerCpu = featureDer.mean({0}).to(torch::kCPU).contiguous();
            auto meanDers = batchDerCpu.data<float>();
            const int fCount = featureDer.size(1);
            out_ << i << " " << "derSigns" << std::endl;
            for (int i = 0; i < fCount; ++i) {
                out_<< (meanDers[i] > 0 ? 1 : -1) << " ";
            }
            out_ << std::endl;
            out_ << i << " " << "meanDers" << std::endl;
            for (int i = 0; i < fCount; ++i) {
                out_<< meanDers[i] << " ";
            }
            out_ << std::endl;
            ++i;
        }

    private:
        int i = 0;
        std::ofstream out_;
    };
}

PolynomCuda::PolynomCuda(PolynomPtr polynom)
: Polynom_(polynom) {

    std::sort(Polynom_->Ensemble_.begin(), Polynom_->Ensemble_.end(), [&](const MonomPtr& left, const MonomPtr& right) {
        return left->Structure_.GetDepth() < right->Structure_.GetDepth();
    });

    std::vector<int> flatFeatureIds;
    std::vector<float> conditions;
    std::vector<int>  offsets;
    std::vector<float> values;
    std::vector<int> origFIds;

    int cursor = 0;
    for (const auto& monom : Polynom_->Ensemble_) {
        for (const auto& split : monom->Structure_.Splits) {
            flatFeatureIds.push_back(split.Feature);
            conditions.push_back(split.Condition);
        }
        values.insert(values.end(), monom->Values_.begin(), monom->Values_.end());
        offsets.push_back(cursor);
        cursor += monom->Structure_.Splits.size();
        origFIds.push_back(monom->origFId_);
    }

    offsets.push_back(cursor);

    Features = Buffer<int>::fromVector(flatFeatureIds).data().to(torch::kCUDA);
    Conditions = Buffer<float>::fromVector(conditions).data().to(torch::kCUDA);
    PolynomOffsets = Buffer<int>::fromVector(offsets).data().to(torch::kCUDA);
    PolynomValues = Buffer<float>::fromVector(values).data().to(torch::kCUDA);
    OrigFIds = Buffer<int>::fromVector(origFIds).data().to(torch::kCUDA);
}


torch::Tensor PolynomCuda::Forward(torch::Tensor batch) const {
    batch = batch.contiguous();
    const int batchSize = batch.size(0);
    int fCount = batch.size(1);

    const int outDim = Polynom_->OutDim();
    const int polynomCount = PolynomOffsets.size(0) - 1;

    torch::Tensor result = torch::zeros({outDim, batchSize},
        TorchHelpers::tensorOptionsOnDevice(ComputeDeviceType::Gpu));

    torch::Tensor probs = torch::zeros({polynomCount, batchSize},
        TorchHelpers::tensorOptionsOnDevice(ComputeDeviceType::Gpu));

    auto transposed = batch.transpose(0, 1).contiguous();

    cudaError_t errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess && errorCode != cudaErrorCudartUnloading) {
        throw std::runtime_error("CUDA error " + std::to_string((int)errorCode) + ": " + cudaGetErrorString(errorCode));
    }
    if (Polynom_->getMonomType() == Monom::MonomType::SigmoidProbMonom) {
        SigmoidProbPolynomForward(Polynom_->Lambda_,
                       transposed.data<float>(),
                       fCount,
                       batchSize,
                       Features.data<int>(),
                       Conditions.data<float>(),
                       PolynomOffsets.data<int>(),
                       PolynomValues.data<float>(),
                       polynomCount,
                       outDim,
                       probs.data<float>(),
                       result.data<float>()
        );
    } else if (Polynom_->getMonomType() == Monom::MonomType::ExpProbMonom) {
        ExpProbPolynomForward(Polynom_->Lambda_,
                                  transposed.data<float>(),
                                  fCount,
                                  batchSize,
                                  Features.data<int>(),
                                  Conditions.data<float>(),
                                  PolynomOffsets.data<int>(),
                                  PolynomValues.data<float>(),
                                  polynomCount,
                                  outDim,
                                  probs.data<float>(),
                                  result.data<float>()
        );
    } else if (Polynom_->getMonomType() == Monom::MonomType::LinearMonom) {
        LinearPolynomForward(Polynom_->Lambda_,
                             transposed.data_ptr<float>(),
                             fCount,
                             batchSize,
                             Features.data_ptr<int>(),
                             Conditions.data_ptr<float>(),
                             PolynomOffsets.data_ptr<int>(),
                             PolynomValues.data_ptr<float>(),
                             polynomCount,
                             outDim,
                             probs.data_ptr<float>(),
                             result.data_ptr<float>(),
                             OrigFIds.data_ptr<int>()
        );
    } else {
        throw std::runtime_error("Unsupported monom type");
    }
    errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess && errorCode != cudaErrorCudartUnloading) {
        throw std::runtime_error("CUDA error " + std::to_string((int)errorCode) + ": " + cudaGetErrorString(errorCode));
    }

    return result.transpose(0, 1).contiguous();

}



torch::Tensor PolynomCuda::Backward(torch::Tensor batch,
                                    torch::Tensor outputDer) const {
    const int batchSize = batch.size(0);
    int fCount = batch.size(1);
    outputDer = outputDer.to(torch::kCUDA).contiguous();
    batch = batch.to(torch::kCUDA).contiguous();
    auto outDim = Polynom_->OutDim();
    const int polynomCount = PolynomOffsets.size(0) - 1;

    VERIFY(outDim == outputDer.size(1), "error: out dim should be equal to polynom out dim");
    torch::Tensor result = torch::zeros({batchSize, fCount},
        TorchHelpers::tensorOptionsOnDevice(ComputeDeviceType::Gpu));

    cudaError_t errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess && errorCode != cudaErrorCudartUnloading) {
        throw std::runtime_error("CUDA error " + std::to_string((int)errorCode) + ": " + cudaGetErrorString(errorCode));
    }
    if (Polynom_->getMonomType() == Monom::MonomType::SigmoidProbMonom) {
        SigmoidProbPolynomBackward(
                batchSize,
                Polynom_->Lambda_,
                batch.data<float>(),
                fCount,
                outputDer.data<float>(),
                outDim,
                Features.data<int>(),
                Conditions.data<float>(),
                PolynomValues.data<float>(),
                PolynomOffsets.data<int>(),
                polynomCount,
                result.data<float>());
    } else if (Polynom_->getMonomType() == Monom::MonomType::ExpProbMonom) {
        ExpProbPolynomBackward(
                batchSize,
                Polynom_->Lambda_,
                batch.data<float>(),
                fCount,
                outputDer.data<float>(),
                outDim,
                Features.data<int>(),
                Conditions.data<float>(),
                PolynomValues.data<float>(),
                PolynomOffsets.data<int>(),
                polynomCount,
                result.data<float>());
    } else if (Polynom_->getMonomType() == Monom::MonomType::LinearMonom) {
        LinearPolynomBackward(
                batchSize,
                Polynom_->Lambda_,
                batch.data_ptr<float>(),
                fCount,
                outputDer.data_ptr<float>(),
                outDim,
                Features.data_ptr<int>(),
                Conditions.data_ptr<float>(),
                PolynomValues.data_ptr<float>(),
                PolynomOffsets.data_ptr<int>(),
                polynomCount,
                result.data_ptr<float>(),
                OrigFIds.data_ptr<int>());
    } else {
        throw std::runtime_error("Unsupported monom type");
    }
    errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess && errorCode != cudaErrorCudartUnloading) {
        throw std::runtime_error("CUDA error " + std::to_string((int)errorCode) + ": " + cudaGetErrorString(errorCode));
    }

    Singleton<PolynomForwardBackwardLog>().invokeBackward(batch, result);

    return result;
}
