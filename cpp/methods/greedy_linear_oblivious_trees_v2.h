#pragma once

#include <unordered_set>
#include <vector>
#include <memory>

#include "optimizer.h"
#include "correlation_bin_stats.h"

#include <models/model.h>
#include <models/bin_optimized_model.h>

#include <data/grid.h>

#include <core/multi_dim_array.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
//#include <eigen3/unsupported/Eigen/CXX11/Tensor>



class GreedyLinearObliviousTreeLearnerV2;




class HistogramV2 {
public:
    HistogramV2(BinarizedDataSet& bds, GridPtr grid, unsigned int nUsedFeatures, int lastUsedFeatureId,
            BinStat* hist);

//    void addFullCorrelation(int bin, Vec x, double y);
    void addNewCorrelation(int bin, const float* xtx, float xty, int shift = 0);
    void prefixSumBins();

    void addBinStat(int bin, const BinStat& stats);

    std::pair<double, double> splitScore(int fId, int condId, double l2reg, double traceReg);

    std::shared_ptr<Eigen::MatrixXf> getW(double l2reg);

    void printEig(double l2reg);
    void printCnt();
    void print();

    HistogramV2& operator+=(const HistogramV2& h);
    HistogramV2& operator-=(const HistogramV2& h);

private:
    static double computeScore(Eigen::MatrixXf& XTX, Eigen::MatrixXf& XTy, double XTX_trace, uint32_t cnt, double l2reg,
                               double traceReg);

    static void printEig(Eigen::MatrixXf& M);

    friend HistogramV2 operator-(const HistogramV2& lhs, const HistogramV2& rhs);
    friend HistogramV2 operator+(const HistogramV2& lhs, const HistogramV2& rhs);

private:
    BinarizedDataSet& bds_;
    GridPtr grid_;

    BinStat* hist_;

    int lastUsedFeatureId_ = -1;
    unsigned int nUsedFeatures_;



    friend class GreedyLinearObliviousTreeLearnerV2;
};

class LinearObliviousTreeLeafV2;

class GreedyLinearObliviousTreeLearnerV2 final
        : public Optimizer {
public:
    // Please, don't even ask me why...
    GreedyLinearObliviousTreeLearnerV2(GridPtr grid, int32_t maxDepth = 6, int biasCol = -1,
                                              double l2reg = 0.0, double traceReg = 0.0)
            : grid_(std::move(grid))
            , biasCol_(biasCol)
            , maxDepth_(maxDepth)
            , l2reg_(l2reg)
            , traceReg_(traceReg)
            , totalBins_(grid_->totalBins())
            , fCount_(grid_->nzFeaturesCount())
            , totalCond_(totalBins_ - fCount_)
            , binOffsets_(grid_->binOffsets())
            , nThreads_((int)GlobalThreadPool<0>().numThreads())
            , h_XTX_(torch::zeros({nThreads_, 1 << maxDepth_, totalBins_, maxDepth_ + 2}, torch::kFloat))
            , h_XTy_(torch::zeros({nThreads_, 1 << maxDepth_, totalBins_}, torch::kFloat))
            , statsData_XTX_(torch::zeros({nThreads_ + 2, 1 << maxDepth_, totalBins_, (maxDepth_ + 2) * (maxDepth_ + 3) / 2}, torch::kFloat))
            , statsData_XTy_(torch::zeros({nThreads_ + 2, 1 << maxDepth_, totalBins_, maxDepth_ + 2}, torch::kFloat))
            , statsData_cnt_(torch::zeros({nThreads_ + 2, 1 << maxDepth_, totalBins_}, torch::kInt))
            , h_XTX_ref_(h_XTX_.accessor<float, 4>())
            , h_XTy_ref_(h_XTy_.accessor<float, 3>())
            , statsData_XTX_ref_(statsData_XTX_.accessor<float, 4>())
            , statsData_XTy_ref_(statsData_XTy_.accessor<float, 4>())
            , statsData_cnt_ref_(statsData_cnt_.accessor<int, 3>()) {

    }

    GreedyLinearObliviousTreeLearnerV2(const GreedyLinearObliviousTreeLearnerV2& other) = default;

    ModelPtr fit(const DataSet& dataSet, const Target& target) override;

private:
    void cacheDs(const DataSet& ds);
    void resetState();

private:
    GridPtr grid_;
    int32_t maxDepth_ = 6;
    int biasCol_ = -1;
    double l2reg_ = 0.0;
    double traceReg_ = 0.0;

    bool isDsCached_ = false;
    std::vector<Vec> fColumns_;
    std::vector<ConstVecRef<float>> fColumnsRefs_;
    std::vector<std::vector<float>> curX_;

    std::vector<int32_t> leafId_;

    std::set<int> usedFeatures_;
    std::vector<int> usedFeaturesOrdered_;

    std::vector<bool> fullUpdate_;
    std::vector<int> samplesLeavesCnt_;

    ConstVecRef<int32_t> binOffsets_;
    int nThreads_;
    int totalBins_;
    int fCount_;
    int totalCond_;
    int nSamples_;

    // thread      leaf         bin         coordinate
//    std::vector<std::vector<std::vector<std::vector<float>>>> h_XTX_;
//    std::vector<std::vector<std::vector<float>>> h_XTy_;
//    std::vector<std::vector<std::vector<BinStat>>> stats_;

    // This one are used to update new correlations
    torch::Tensor h_XTX_;
    torch::Tensor h_XTy_;

    // This ones are used in bin stats
    torch::Tensor statsData_XTX_;
    torch::Tensor statsData_XTy_;
    torch::Tensor statsData_cnt_;

    torch::TensorAccessor<float, 4> h_XTX_ref_;
    torch::TensorAccessor<float, 3> h_XTy_ref_;
    torch::TensorAccessor<float, 4> statsData_XTX_ref_;
    torch::TensorAccessor<float, 4> statsData_XTy_ref_;
    torch::TensorAccessor<int, 3> statsData_cnt_ref_;

    unsigned int curLeavesCoord_;

    std::vector<BinStat> statsV_;
    multi_dim_array_idxs statsIdxs_;
    MultiDimArray<3, BinStat> stats_;
};

class LinearObliviousTreeV2 final
        : public Stub<BinOptimizedModel, LinearObliviousTreeV2>
        , std::enable_shared_from_this<LinearObliviousTreeV2> {
public:

    LinearObliviousTreeV2(const LinearObliviousTreeV2& other, double scale)
            : Stub<BinOptimizedModel, LinearObliviousTreeV2>(other.gridPtr()->origFeaturesCount(), 1) {
        grid_ = other.grid_;
        scale_ = scale;
        leaves_ = other.leaves_;
        splits_ = other.splits_;
    }

    LinearObliviousTreeV2(GridPtr grid, std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> leaves)
            : Stub<BinOptimizedModel, LinearObliviousTreeV2>(grid->origFeaturesCount(), 1)
            , grid_(std::move(grid))
            , leaves_(std::move(leaves)) {
        scale_ = 1;
    }

    explicit LinearObliviousTreeV2(GridPtr grid)
            : Stub<BinOptimizedModel, LinearObliviousTreeV2>(grid->origFeaturesCount(), 1)
            , grid_(std::move(grid)) {

    }

    Grid grid() const {
        return *grid_.get();
    }

    GridPtr gridPtr() const {
        return grid_;
    }

    void appendTo(const Vec& x, Vec to) const override;

    void applyToBds(const BinarizedDataSet& ds, Mx to, ApplyType type) const;

    void applyBinarizedRow(const Buffer<uint8_t>& x, Vec to) const {
        throw std::runtime_error("Unsupported");
    }

    double value(const Vec& x) override;

    void grad(const Vec& x, Vec to) override;

private:
    friend class GreedyLinearObliviousTreeLearnerV2;

    double value(const ConstVecRef<float>& x) const;

private:
    GridPtr grid_;
    double scale_ = 1;

    std::vector<std::tuple<int32_t, int32_t>> splits_;

    std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> leaves_;
};
