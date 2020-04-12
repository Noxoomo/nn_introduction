#pragma once

#include "context.h"
#include <torch/torch.h>


namespace TorchHelpers {

    inline int64_t totalSize(const torch::Tensor& tensor) {
        int64_t size = 1;
        for (auto dimSize : tensor.sizes()) {
            size *= dimSize;
        }
        return size;
    }

    inline torch::Device torchDevice(const ComputeDevice& device) {
        switch (device.deviceType()) {
            case ComputeDeviceType::Cpu: {
                return torch::Device(torch::DeviceType::CPU);
            }
            case ComputeDeviceType::Gpu: {
                return torch::Device(torch::DeviceType::CUDA);
            }
            default: {
                assert(false);
            }
        }
    }

    inline torch::TensorOptions tensorOptionsOnDevice(
        const ComputeDevice device,
        torch::ScalarType dtype = torch::ScalarType::Float) {
        torch::TensorOptions baseOptions = [&]() {
            return torch::device(torchDevice(device));
        }();
        baseOptions = baseOptions.requires_grad(false);
        baseOptions = baseOptions.dtype(dtype);
        return baseOptions;
    }


    inline ComputeDevice getDevice(const torch::Tensor& tensor) {
        if (tensor.is_cuda()) {
            return ComputeDevice(ComputeDeviceType::Gpu);
        } else {
            assert(tensor.device() == c10::kCPU);
            return ComputeDevice(ComputeDeviceType::Cpu);
        }
    }

}
