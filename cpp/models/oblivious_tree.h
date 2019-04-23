#pragma once

#include "bin_optimized_model.h"
#include <data/grid.h>
#include <core/vec_factory.h>
#include <core/func.h>


class ObliviousTree final : public Stub<BinOptimizedModel, ObliviousTree> {
public:

    ObliviousTree(GridPtr grid,
                  std::vector<BinaryFeature> binFeatures,
                  Vec leaves
                  )
      : Stub<BinOptimizedModel, ObliviousTree>(grid->origFeaturesCount(), 1)
      , grid_(std::move(grid))
      , splits_(std::move(binFeatures))
      , leaves_(leaves) {
            bitVec.reserve(leaves_.size());
            auto leavesPtr = leaves_.arrayRef();
            for (uint32_t b = 0; b < leaves_.size(); ++b) {
                double value = 0;
                uint bitsB = bits(b);
                for (uint a = 0; a < leaves_.size(); ++a) {
                    uint bitsA = bits(a);
                    if (bits(a & b) >= bitsA)
                        value += (((bitsA + bitsB) & 1) > 0 ? -1 : 1) * leavesPtr[a];
                }
                bitVec[b] = value;
            }
    }


    ObliviousTree(const ObliviousTree& other, double scale = 1.0)
    : Stub<BinOptimizedModel, ObliviousTree>(other)
    , grid_(other.grid_)
    , splits_(other.splits_)
    , leaves_(scale == 1.0  ? other.leaves_ : VecFactory::clone(other.leaves_) * scale) {
        bitVec.reserve(leaves_.size());
        auto leavesPtr = leaves_.arrayRef();

        for (uint32_t b = 0; b < leaves_.size(); ++b) {
            double value = 0;
            uint bitsB = bits(b);
            for (uint a = 0; a < leaves_.size(); ++a) {
                uint bitsA = bits(a);
                if (bits(a & b) >= bitsA)
                    value += (((bitsA + bitsB) & 1) > 0 ? -1 : 1) * leavesPtr[a];
            }
            bitVec[b] = value;
        }
    }

    const Grid& grid() const {
        return *grid_;
    }

    GridPtr gridPtr() const override {
        return grid_;
    }

    void appendTo(const Vec& x, Vec to) const override;

    void applyToBds(const BinarizedDataSet& ds, Mx to, ApplyType type) const override;

    void applyBinarizedRow(const Buffer<uint8_t>& x, Vec to) const;

    double value(const Vec& x);

    void grad(const Vec& x, Vec to);

private:
    uint32_t bits(uint32_t i);


private:
    GridPtr grid_;
    std::vector<BinaryFeature> splits_;
    Vec leaves_;
    std::vector<double> bitVec;
};
