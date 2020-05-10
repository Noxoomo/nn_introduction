#pragma once

#include <memory>
#include <stdexcept>

#include "bin_optimized_model.h"
#include <data/grid.h>
#include <core/vec_factory.h>
#include <core/func.h>
#include <util/io.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

struct LinearObliviousTreeLeaf : std::enable_shared_from_this<LinearObliviousTreeLeaf> {
public:
    LinearObliviousTreeLeaf() = default;

    LinearObliviousTreeLeaf(
            std::vector<int32_t> usedFeaturesInOrder,
            Eigen::MatrixXd w,
            double weight)
            : usedFeaturesInOrder_(std::move(usedFeaturesInOrder))
            , w_(std::move(w))
            , weight_(weight) {
    }

    double value(const ConstVecRef<float>& x) const {
        double res = 0.0;

        res += w_(0, 0); // bias
        for (int i = 1; i < (int)w_.size(); ++i) {
            int f = usedFeaturesInOrder_[i];
            res += x[f] * w_(i, 0);
        }

        return res;
    }

    void grad(VecRef<float> to, double scale) {
        int i = 0;
        for (int f : usedFeaturesInOrder_) {
            // TODO we treat f = -1 as bias
            if (f != -1) {
                to[f] += w_(i, 0) * scale;
            }
            ++i;
        }
    }

    void printInfo(double scale = 1.0) const {
        std::cout << "{";
        for (int i = 0; i < w_.size(); ++i) {
            std::cout << std::setprecision(7) << w_(i, 0) * scale;
            if (i != w_.size() - 1) {
                std::cout << " ";
            }
        }
        std::cout << "}@" << weight_;
    }

    void serialize(std::ostream& out) const {
        std::size_t size = usedFeaturesInOrder_.size();
        out.write("l{", 2);

        out.write("s{", 2);
        out.write((char *) &size, sizeof(size));
        out.write("}", 1);

        out.write((char *) usedFeaturesInOrder_.data(), sizeof(int32_t) * size);
        out.write((char *) w_.data(), sizeof(double) * size);
        out.write((char *) &weight_, sizeof(weight_));

        out.write("}", 1);
    }

    static bool deserialize(std::istream& in, LinearObliviousTreeLeaf& leaf) {
        if (!checkStrPresent(in, "l{")) {
            return false;
        }

        std::size_t size;
        if (!couldRead(in, &size, sizeof(size), "s")) {
            return false;
        }

        std::vector<int32_t> usedFeaturesInOrder(size);
        in.read((char *)usedFeaturesInOrder.data(), sizeof(int32_t) * size);

        Eigen::MatrixXd w(size, 1);
        in.read((char*)w.data(), sizeof(double) * size);

        double weight;
        in.read((char*)&weight, sizeof(weight));

        if (!checkStrPresent(in, "}")) {
            return false;
        }

        leaf.usedFeaturesInOrder_ = std::move(usedFeaturesInOrder);
        leaf.w_ = std::move(w);
        leaf.weight_ = weight;

        return true;
    }

    friend class GreedyLinearObliviousTreeLearner;

    std::vector<int32_t> usedFeaturesInOrder_;
    Eigen::MatrixXd w_;
    double weight_;
};

// TODO separate into two: one is pure inference, without a grid, another is BinOptimized and stores grid
struct LinearObliviousTree final
        : public Stub<BinOptimizedModel, LinearObliviousTree>
        , std::enable_shared_from_this<LinearObliviousTree> {
public:

    LinearObliviousTree(const LinearObliviousTree& other, double scale)
            : Stub<BinOptimizedModel, LinearObliviousTree>(other.gridPtr()->origFeaturesCount(), 1) {
        grid_ = other.grid_;
        xdim_ = other.xdim_;
        ydim_ = other.ydim_;
        scale_ = scale;
        leaves_ = other.leaves_;
        splits_ = other.splits_;
    }

    LinearObliviousTree(GridPtr grid, std::vector<LinearObliviousTreeLeaf> leaves)
            : Stub<BinOptimizedModel, LinearObliviousTree>(grid->origFeaturesCount(), 1)
            , grid_(std::move(grid))
            , xdim_(grid->origFeaturesCount())
            , ydim_(1)
            , leaves_(std::move(leaves)) {
        scale_ = 1;
    }

    explicit LinearObliviousTree(GridPtr grid)
            : Stub<BinOptimizedModel, LinearObliviousTree>(grid->origFeaturesCount(), 1)
            , grid_(std::move(grid))
            , xdim_(grid_->origFeaturesCount())
            , ydim_(1) {

    }

    LinearObliviousTree(int xdim, int ydim)
            : Stub<BinOptimizedModel, LinearObliviousTree>(xdim, ydim)
            , xdim_(xdim)
            , ydim_(ydim) {

    }

    Grid grid() const {
        if (!grid_) {
            throw std::runtime_error("No grid");
        }
        return *grid_.get();
    }

    GridPtr gridPtr() const override {
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

    void printInfo() const;

    friend class GreedyLinearObliviousTreeLearner;

    double value(const ConstVecRef<float>& x) const;

    int getLeaf(const ConstVecRef<float>& x) const;

    void serialize(std::ostream& out) const {
        out.write("t{", 2);

        out.write((char*)&xdim_, sizeof(xdim_));
        out.write((char*)&ydim_, sizeof(ydim_));

        out.write((char*)&scale_, sizeof(scale_));

        int splitsSize = splits_.size();
        out.write("s{", 2);
        out.write((char*)&splitsSize, sizeof(splitsSize));
        out.write("}", 1);

        for (const auto& split : splits_) {
            int32_t splitOrigFId = std::get<0>(split);
            double splitCondId = std::get<1>(split);
            out.write((char*)&splitOrigFId, sizeof(splitOrigFId));
            out.write((char*)&splitCondId, sizeof(splitCondId));
        }

        int nLeaves = leaves_.size();
        out.write("n{", 2);
        out.write((char*)&nLeaves, sizeof(nLeaves));
        out.write("}", 1);

        for (const auto& l : leaves_) {
            l.serialize(out);
        }

        out.write("}", 1);
    }

    static std::shared_ptr<LinearObliviousTree> deserialize(std::istream& in, GridPtr grid) {
        if (!checkStrPresent(in, "t{")) {
            return nullptr;
        }

        int xdim;
        if (!in.read((char*)&xdim, sizeof(xdim))) {
            return nullptr;
        }

        int ydim;
        if (!in.read((char*)&ydim, sizeof(ydim))) {
            return nullptr;
        }

        double scale;
        if (!in.read((char*)&scale, sizeof(scale))) {
            return nullptr;
        }

        int splitsSize;
        if (!couldRead(in, &splitsSize, sizeof(splitsSize), "s")) {
            return nullptr;
        }

        std::vector<std::tuple<int32_t, double>> splits;
        for (int i = 0; i < splitsSize; ++i) {
            int32_t splitOrigFId;
            double splitCond;
            if (!in.read((char*)&splitOrigFId, sizeof(splitOrigFId))) {
                return nullptr;
            }
            if (!in.read((char*)&splitCond, sizeof(splitCond))) {
                return nullptr;
            }
            splits.emplace_back(std::make_tuple(splitOrigFId, splitCond));
        }

        int nLeaves;
        if (!couldRead(in, &nLeaves, sizeof(nLeaves), "n")) {
            return nullptr;
        }

        std::vector<LinearObliviousTreeLeaf> leaves;
        for (int i = 0; i < nLeaves; ++i) {
            LinearObliviousTreeLeaf leaf;
            if (!LinearObliviousTreeLeaf::deserialize(in, leaf)) {
                return nullptr;
            }
            leaves.emplace_back(std::move(leaf));
        }

        if (!checkStrPresent(in, "}")) {
            return nullptr;
        }

        std::shared_ptr<LinearObliviousTree> tree;

        if (grid) {
            assert(xdim == grid->origFeaturesCount());
            tree = std::make_shared<LinearObliviousTree>(grid);
        } else {
            std::cout << "without grid" << std::endl;
            tree = std::make_shared<LinearObliviousTree>(xdim, ydim);
        }

        tree->scale_ = scale;
        tree->splits_ = std::move(splits);
        tree->leaves_ = std::move(leaves);

        return tree;
    }

    GridPtr grid_;
    int xdim_ = 1;
    int ydim_ = 1;
    double scale_ = 1;

    std::vector<std::tuple<int32_t, double>> splits_;

    std::vector<LinearObliviousTreeLeaf> leaves_;
};
