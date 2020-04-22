#pragma once

#include "dataset.h"
#include "grid.h"
#include <core/vec.h>
#include <util/array_ref.h>
#include <util/json.h>

enum class GridType {
    GreedyLogSum
};


struct BinarizationConfig {
    GridType type_ = GridType::GreedyLogSum;

    uint32_t bordersCount_ = 32;

    static BinarizationConfig fromJson(const json& params);
};


std::vector<float> buildBorders(const BinarizationConfig& config, Vec* vals);

GridPtr buildGrid(const DataSet& ds, const BinarizationConfig& config);
