#pragma once

#include "experiments/core/tensor_pair_dataset.h"

#include <torch/torch.h>

#include <utility>

namespace experiments::svhn {

std::pair<TensorPairDataset, TensorPairDataset> read_dataset(
        const std::string &folder,
        int training_limit = -1,
        int test_limit = -1);

}
