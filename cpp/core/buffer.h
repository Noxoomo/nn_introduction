#pragma once

#include "torch_helpers.h"
#include <torch/torch.h>
#include <util/array_ref.h>
#include <utility>

namespace Detail {

    template <class T>
    class TorchBufferTrait {
    public:

        static torch::Tensor create(int64_t size) {
            return torch::zeros({static_cast<int64_t>(size * sizeof(T))}, TorchHelpers::tensorOptionsOnDevice(CurrentDevice(),
                                                                                                               torch::ScalarType::Byte));
        }

        static uint8_t* data(const torch::Tensor& tensor) {
            return tensor.data_ptr<uint8_t>();
        }


        static int64_t size(const torch::Tensor& tensor) {
            return TorchHelpers::totalSize(tensor) / sizeof(T);
        }
    };


    //todo: make specialisation based for all specialisations
    template <>
    class TorchBufferTrait<float> {
    public:

        static torch::Tensor create(int64_t size) {
            return torch::zeros({static_cast<int64_t>(size)}, TorchHelpers::tensorOptionsOnDevice(CurrentDevice(),
                                                                                                              torch::ScalarType::Float));
        }

        static float* data(const torch::Tensor& tensor) {
            return tensor.data_ptr<float>();
        }

        static int64_t size(const torch::Tensor& tensor) {
            return TorchHelpers::totalSize(tensor);
        }
    };


    template <>
    class TorchBufferTrait<int> {
    public:

        static torch::Tensor create(int64_t size) {
            return torch::zeros({static_cast<int64_t>(size)}, TorchHelpers::tensorOptionsOnDevice(CurrentDevice(),
                                                                                                  torch::ScalarType::Int));
        }

        static int* data(const torch::Tensor& tensor) {
            return tensor.data_ptr<int>();
        }

        static int64_t size(const torch::Tensor& tensor) {
            return TorchHelpers::totalSize(tensor);
        }
    };
}

template <class T>
class Buffer  {
public:

    //TODO(noxoomo): should not clear memory  and this won't work with multiGPU
    explicit Buffer(int64_t size)
    : data_(Detail::TorchBufferTrait<T>::create(size)) {

    }


    Buffer()
    : Buffer(0) {

    }

    VecRef<T> arrayRef() {
        return VecRef<T>(reinterpret_cast<T*>(Detail::TorchBufferTrait<T>::data(data_)), size());
    }

    void fill(const T& val) {
        auto dst = arrayRef();
        for (int64_t i = 0; i < dst.size(); ++i) {
            dst[i] = val;
        }
    }

    ConstVecRef<T> arrayRef() const {
        return ConstVecRef<T>(reinterpret_cast<const T*>(Detail::TorchBufferTrait<T>::data(data_)), size());
    }


    static Buffer create(int64_t size) {
        return Buffer(Detail::TorchBufferTrait<T>::create(size));
    }

    Buffer copy() const {
        return Buffer(data_.clone());
    }

    int64_t size() const {
        return  Detail::TorchBufferTrait<T>::size(data_);
    }

    void swap(Buffer<T> other) {
        std::swap(data_, other.data_);
    }


    static Buffer fromVector(const std::vector<T>& vec) {
        Buffer x(static_cast<int64_t>(vec.size()));
        VecRef<T> dst = x.arrayRef();
        for (int64_t i = 0; i < dst.size(); ++i) {
            dst[i] = vec[i];
        }
        return x;
    }

    operator const torch::Tensor&() const {
        return data();
    }

    operator torch::Tensor&() {
        return data();
    }

    const torch::Tensor& data() const {
        return data_;
    }

    torch::Tensor& data() {
        return data_;
    }

    bool isContiguous() const {
        return data_.is_contiguous();
    }

    Buffer& operator=(const Buffer& other) {
        data_ = other.data_;
        return *this;
    }

    ComputeDevice device() const {
        return TorchHelpers::getDevice(data());
    }

    bool isCpu() const {
        return device().deviceType() == ComputeDeviceType::Cpu;
    }
protected:

    explicit Buffer(torch::Tensor data)
    : data_(std::move(data)) {

}
protected:
    torch::Tensor data_;
};
