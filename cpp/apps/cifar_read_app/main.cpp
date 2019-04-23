#include "experiments/cifar_nn/cifar10_reader.h"

#include <cstdint>
#include <vector>
#include <iostream>
#include <string>

#include <torch/torch.h>

int main() {
    const std::string& path = "../../../../python/resources/cifar10/cifar-10-batches-bin";
    auto dataset = cifar::read_dataset(path);
    std::cout << "train size: " << dataset.first.size().value() << "\n"
            << "test size: " << dataset.second.size().value() << std::endl;
}
