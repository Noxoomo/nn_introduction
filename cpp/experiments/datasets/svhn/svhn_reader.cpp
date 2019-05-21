#include "svhn_reader.h"

#include <experiments/core/tensor_pair_dataset.h>
#include <experiments/core/transform.h>

#include <torch/torch.h>
#include <torch/script.h>

#include <utility>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

namespace experiments::svhn {

#define ROW_SIZE (3 * 32 * 32)

static TensorPairDataset readDs(const std::string& folder,
        const std::string& file,
        int limit) {
    std::vector<torch::Tensor> xs;
    std::vector<int> ys;

    std::ifstream dsFile;
    dsFile.open(folder + "/" + file);

    for (int i = 0; i < limit; ++i) {
        std::string str;
        std::vector<float> img;
        for (int j = 0; j < ROW_SIZE; ++j) {
            getline(dsFile, str, ',');
            float f = std::stof(str);
            img.push_back(f);
        }
        getline(dsFile, str);
        int y = std::stoi(str);
        ys.push_back(y);

        auto x = torch::tensor(img, torch::kFloat32);
        x = x.view({3, 32, 32});
        xs.push_back(x);
    }

    dsFile.close();

    auto x = torch::stack(xs, 0);
    auto y = torch::tensor(ys, torch::kLong);

    return {x, y};
}

// Note that we store svhn dataset already normalized

std::pair<TensorPairDataset, TensorPairDataset> read_dataset(
        const std::string &folder,
        int training_limit,
        int test_limit) {
    if (training_limit == -1) {
        training_limit = 73257;
    }

    if (test_limit == -1) {
        test_limit = 26032;
    }

    auto trainDs = readDs(folder, "train_ds", training_limit);
    auto testDs = readDs(folder, "test_ds", test_limit);

    return std::make_pair(trainDs, testDs);
}

}
