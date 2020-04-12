#pragma once

#include "dataset.h"
#include "grid.h"
#include <core/vec.h>
#include <util/array_ref.h>

enum class GridType {
    GreedyLogSum
};


struct BinarizationConfig {
    GridType type_ = GridType::GreedyLogSum;

    uint32_t bordersCount_ = 32;
};


std::vector<float> buildBorders(const BinarizationConfig& config, Vec* vals);

GridPtr buildGrid(const DataSet& ds, const BinarizationConfig& config);
