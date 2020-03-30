#pragma once

#include <memory>

#include "bin_optimized_model.h"
#include <data/grid.h>
#include <core/vec_factory.h>
#include <core/func.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

class LinearObliviousTreeLeaf : std::enable_shared_from_this<LinearObliviousTreeLeaf> {
public:
    LinearObliviousTreeLeaf(
            std::vector<int32_t> usedFeaturesInOrder,
            Eigen::MatrixXd w)
            : usedFeaturesInOrder_(std::move(usedFeaturesInOrder))
            , w_(std::move(w)) {
    }

    double value(const ConstVecRef<float>& x) const {
        float res = 0.0;

        int i = 0;
        for (auto f : usedFeaturesInOrder_) {
            res += x[f] * w_(i, 0);
            ++i;
        }

        return res;
    }

private:
    friend class GreedyLinearObliviousTreeLearner;

    std::vector<int32_t> usedFeaturesInOrder_;
    Eigen::MatrixXd w_;
};

class LinearObliviousTree final
        : public Stub<BinOptimizedModel, LinearObliviousTree>
        , std::enable_shared_from_this<LinearObliviousTree> {
public:

    LinearObliviousTree(const LinearObliviousTree& other, double scale)
            : Stub<BinOptimizedModel, LinearObliviousTree>(other.gridPtr()->origFeaturesCount(), 1) {
        grid_ = other.grid_;
        scale_ = scale;
        leaves_ = other.leaves_;
        splits_ = other.splits_;
    }

    LinearObliviousTree(GridPtr grid, std::vector<LinearObliviousTreeLeaf> leaves)
            : Stub<BinOptimizedModel, LinearObliviousTree>(grid->origFeaturesCount(), 1)
            , grid_(std::move(grid))
            , leaves_(std::move(leaves)) {
        scale_ = 1;
    }

    explicit LinearObliviousTree(GridPtr grid)
            : Stub<BinOptimizedModel, LinearObliviousTree>(grid->origFeaturesCount(), 1)
            , grid_(std::move(grid)) {

    }

    Grid grid() const {
        return *grid_.get();
    }

    GridPtr gridPtr() const {
        return grid_;
    }

    // I have now idea what this function should do...
    // For now just adding value(x) to @param to.
    void appendTo(const Vec& x, Vec to) const override;

    void applyToBds(const BinarizedDataSet& ds, Mx to, ApplyType type) const override;

    void applyBinarizedRow(const Buffer<uint8_t>& x, Vec to) const {
        throw std::runtime_error("Unimplemented");
    }

    double value(const Vec& x) override;

    void grad(const Vec& x, Vec to) override;

private:
    friend class GreedyLinearObliviousTreeLearner;

    double value(const ConstVecRef<float>& x) const;

private:
    GridPtr grid_;
    double scale_ = 1;

    std::vector<std::tuple<int32_t, int32_t>> splits_;

    std::vector<LinearObliviousTreeLeaf> leaves_;
};
