#pragma once
//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains functions to read the CIFAR-10 dataset
 */


#include "tensor_pair_dataset.h"

#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <utility>
#include <cstdint>
#include <chrono>

namespace cifar {

#define CIFAR10_FILE_SIZE 10000
#define IMAGE_SIZE (3 * 32 * 32)

    std::unique_ptr<uint8_t[]> read_cifar10_file_buffer(const std::string& path);

    void read_cifar10_file_float(float* x, long* y, const std::string& path, int& limit);

/*!
 * \brief Read all test data.
 *
 * The dataset is assumed to be in a cifar-10 subfolder
 *
 * \param limit The maximum number of elements to read (0: no limit)
 */
    void read_test(const std::string& folder, int limit, float* images, long* labels);

/*!
 * \brief Read all training data
 *
 * The dataset is assumed to be in a cifar-10 subfolder
 *
 * \param limit The maximum number of elements to read (0: no limit)
 */
    void read_training(const std::string& folder, int limit, float* images, long* labels);

    std::pair<TensorPairDataset, TensorPairDataset> read_dataset(
        const std::string& folder,
        int training_limit = -1,
        int test_limit = -1);

} //end of namespace cifar

