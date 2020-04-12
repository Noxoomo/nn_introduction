#include "grid_builder.h"
#include <core/vec_factory.h>
#include <vec_tools/sort.h>
#include <util/exception.h>

namespace {

    template <class It>
    class FeatureBin {
    private:
        uint32_t BinStart;
        uint32_t BinEnd;
        It FeaturesStart;
        It FeaturesEnd;

        uint32_t BestSplit;
        double BestScore;

        inline void UpdateBestSplitProperties() {
            const int mid = (BinStart + BinEnd) / 2;
            auto midValue = *(FeaturesStart + mid);

            uint32_t lb = (std::lower_bound(FeaturesStart + BinStart, FeaturesStart + mid, midValue) - FeaturesStart);
            uint32_t up = (std::upper_bound(FeaturesStart + mid, FeaturesStart + BinEnd, midValue) - FeaturesStart);

            const double scoreLeft = lb != BinStart ? log((double)(lb - BinStart)) + log((double)(BinEnd - lb)) : 0.0;
            const double scoreRight = up != BinEnd ? log((double)(up - BinStart)) + log((double)(BinEnd - up)) : 0.0;
            BestSplit = scoreLeft >= scoreRight ? lb : up;
            BestScore = BestSplit == lb ? scoreLeft : scoreRight;
        }

    public:
        FeatureBin(uint32_t binStart, uint32_t binEnd, It featuresStart, It featuresEnd)
            : BinStart(binStart)
              , BinEnd(binEnd)
              , FeaturesStart(featuresStart)
              , FeaturesEnd(featuresEnd)
              , BestSplit(BinStart)
              , BestScore(0.0){
            UpdateBestSplitProperties();
        }

        uint32_t Size() const {
            return BinEnd - BinStart;
        }

        bool operator<(const FeatureBin& bf) const {
            return Score() < bf.Score();
        }

        double Score() const {
            return BestScore;
        }

        FeatureBin Split() {
            if (!CanSplit()) {
                throw Exception() << "Can't add new split";
            }
            FeatureBin left = FeatureBin(BinStart, BestSplit, FeaturesStart, FeaturesEnd);
            BinStart = BestSplit;
            UpdateBestSplitProperties();
            return left;
        }

        bool CanSplit() const {
            return (BinStart != BestSplit && BinEnd != BestSplit);
        }

        float Border() const {
            assert(BinStart < BinEnd);
            double borderValue = 0.5f * (*(FeaturesStart + BinEnd - 1));
            const double nextValue = ((FeaturesStart + BinEnd) < FeaturesEnd)
                                    ? (*(FeaturesStart + BinEnd))
                                    : (*(FeaturesStart + BinEnd - 1));
            borderValue += 0.5f * nextValue;
            return static_cast<float>(borderValue);
        }

        bool IsLast() const {
            return BinEnd == (FeaturesEnd - FeaturesStart);
        }
    };
}


std::vector<float> buildBorders(const BinarizationConfig& config, Vec* vals) {
    std::vector<float> borders;
    if (vals->dim()) {
        auto sortedFeature = VecFactory::toDevice(VecTools::sort(*vals), ComputeDevice(ComputeDeviceType::Cpu));
        auto data = sortedFeature.arrayRef();
        for (int32_t i = 1; i < data.size(); ++i) {
            assert(data[i] >= data[i - 1]);
        }
        const uint32_t dim = static_cast<uint32_t>(sortedFeature.dim());
        using It = decltype(data.begin());
        std::priority_queue<FeatureBin<It>> splits;
        splits.push(FeatureBin(0, dim, data.begin(), data.end()));

        while (splits.size() <= (uint32_t) config.bordersCount_ && splits.top().CanSplit()) {
            FeatureBin top = splits.top();
            splits.pop();
            splits.push(top.Split());
            splits.push(top);
        }

        while (!splits.empty()) {
            if (!splits.top().IsLast()) {
                borders.push_back(splits.top().Border());
            }
            splits.pop();
        }
    }
    std::sort(borders.begin(), borders.end());
    return borders;
}



GridPtr buildGrid(const DataSet& ds, const BinarizationConfig& config) {
    std::vector<BinaryFeature> binFeatures;
    std::vector<Feature> features;
    std::vector<std::vector<float>> borders;

    Vec column(ds.samplesCount());

    int32_t nzFeatureId = 0;

    for (int32_t fIndex = 0; fIndex < ds.featuresCount(); ++fIndex) {
        ds.copyColumn(fIndex, &column);

        auto featureBorders = buildBorders(config, &column);
        if (!featureBorders.empty()) {
            borders.push_back(featureBorders);
            const auto binCount = borders.back().size();
            features.push_back(Feature(nzFeatureId, binCount, fIndex));
            for (int32_t bin = 0; bin < binCount; ++bin) {
                binFeatures.emplace_back(nzFeatureId, bin);
            }
            ++nzFeatureId;
        }

    }

    return std::make_shared<Grid>(
        ds.featuresCount(),
        std::move(binFeatures),
        std::move(features),
        std::move(borders));
}

