#include "binarized_dataset.h"


BinarizedDataSetPtr binarize(const DataSet& ds, GridPtr& gridPtr, int32_t maxGroupSize) {
    const auto& grid = *gridPtr;
    std::unique_ptr<BinarizedDataSet>
        bds(new BinarizedDataSet(ds, gridPtr, ds.samplesCount(), createGroups(grid, maxGroupSize)));

    const int64_t groups = bds->groupCount();

    std::vector<uint8_t> binarizedLine(grid.nzFeaturesCount());

    for (int64_t line = 0; line < ds.samplesCount(); ++line) {
        const Vec lineFeatures = ds.sample(line);
        ConstVecRef<float> row = lineFeatures.arrayRef();
        grid.binarize(row, binarizedLine);

        for (int64_t i = 0; i < groups; ++i) {
            bds->updateLineForGroup(i, line, [&](ConstVecRef<int32_t> gridFeatures, VecRef<uint8_t> bins) {
                for (int64_t f = 0; f < gridFeatures.size(); ++f) {
                    bins[f] = binarizedLine[gridFeatures[f]];
                }
            });
        }
    }

    return bds;
}

void createGroups(
    const Grid& grid,
    int32_t maxGroupSize,
    std::vector<FeaturesBundle>* bundles) {


    FeaturesBundle cursor;
    int32_t byteIdx = 0;


    //TODO: dynamaic packging (32, 16, 8 , 1) to account for last features
    for (int32_t fIndex = 0; fIndex < grid.nzFeaturesCount(); ++fIndex) {
        cursor.lastFeature_++;
        ++byteIdx;

        if (cursor.groupSize() >= maxGroupSize || cursor.lastFeature_ == grid.nzFeaturesCount()) {
            bundles->push_back(cursor);
            cursor = FeaturesBundle();
            cursor.firstFeature_ = bundles->back().lastFeature_;
            cursor.lastFeature_ = bundles->back().lastFeature_;
            cursor.groupOffset_ = byteIdx;
        }
    }
}

