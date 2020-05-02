#include "grid_builder.h"
#include <core/vec_factory.h>
#include <vec_tools/sort.h>
#include <util/exception.h>

namespace {

    // Borders for features.txt taken from jmll
    // I just leave it here for now for experiments
    static std::vector<std::vector<float>> jmllFBorders = {
/* bias */          {},
/*  0 */            {0.0,        0.00252695, 0.0488544,  0.109522,    0.123526,   0.134959,   0.146291,   0.156937,   0.165865,   0.174794,   0.184238,   0.192995,   0.202095,   0.210508,   0.219093,   0.228709,   0.238496,   0.248111,   0.258929,   0.27112,    0.285371,   0.29842,    0.312842,  0.331164,  0.351541,  0.376984,  0.419478,  0.477206,  0.563471,  0.686339,  0.825281},
/*  1 */            {0.0,        0.00189394, 0.00405844, 0.00730519,  0.0113095,  0.0135823,  0.015855,   0.0205628,  0.0261364,  0.0323052,  0.0399351,  0.0479978,  0.0568723,  0.0676707,  0.0786145,  0.0902108,  0.103062,   0.11747,    0.134191,   0.151976,   0.173805,   0.196948,   0.227172,  0.259947,  0.297399,  0.346287,  0.404771,  0.469973,  0.550152,  0.664677,  0.806501},
/*  2 */            {0.0,        0.00275735, 0.00574449, 0.00861673,  0.0128676,  0.0168888,  0.0219439,  0.0283778,  0.0358456,  0.0451517,  0.0549173,  0.0651203,  0.0787073,  0.0921972,  0.107143,   0.12648,    0.153125,   0.178372,   0.21064,    0.249063,   0.293153,   0.342288,   0.397845,  0.453923,  0.513426,  0.58319,   0.649175,  0.730013,  0.812986,  0.879324,  0.937755},
/*  3 */            {0.0},
/*  4 */            {0.0},
/*  5 */            {0.0},
/*  6 */            {0.0},
/*  7 */            {0.0},
/*  8 */            {0.0},
/*  9 */            {0.0},
/* 10 */            {0.0},
/* 11 */            {0.0},
/* 12 */            {0.0},
/* 13 */            {0.0},
/* 14 */            {0.0,        0.321569,   0.341176,   0.360784,    0.376471,   0.392157,   0.4,        0.407843,   0.419608,   0.427451,   0.439216,   0.447059,   0.45098,    0.458824,   0.466667,   0.47451,    0.486275,   0.494118,   0.505882,   0.513726,   0.521569,   0.533333,   0.545098,  0.556863,  0.568627,  0.584314,  0.6,       0.627451,  0.65098,   0.686275,  0.737255},
/* 15 */            {0.259259,   0.310345,   0.354839,   0.393939,    0.428571,   0.459459,   0.487179,   0.52381,    0.545455,   0.574468,   0.591837,   0.615385,   0.636364,   0.661017,   0.68254,    0.701493,   0.714286,   0.733333,   0.746835,   0.761905,   0.78022,    0.79798,    0.813084,  0.831933,  0.847328,  0.863946,  0.880952,  0.897436,  0.916318,  0.937304,  0.962121},
/* 16 */            {0.0,        0.0132378,  0.0182292,  0.0210503,   0.0243056,  0.0299479,  0.0360243,  0.0418837,  0.0477431,  0.0546875,  0.062066,   0.0703671,  0.0786713,  0.0889423,  0.0992133,  0.110358,   0.120629,   0.133304,   0.14729,    0.161932,   0.177885,   0.195367,   0.212413,  0.231862,  0.254808,  0.280813,  0.313593,  0.361451,  0.428694,  0.521101,  0.674377},
/* 17 */            {0.0776251,  0.0996135,  0.11682,    0.127417,    0.133137,   0.147705,   0.155208,   0.161797,   0.16986,    0.174969,   0.181279,   0.186173,   0.191978,   0.197963,   0.204841,   0.211367,   0.215811,   0.22051,    0.226066,   0.232311,   0.238515,   0.247334,   0.256972,  0.268045,  0.274809,  0.286098,  0.299465,  0.307966,  0.329018,  0.344844,  0.381329},
/* 18 */            {0.0},
/* 19 */            {0.0},
/* 20 */            {0.0,        0.227451,   0.239216,   0.254902,    0.27451,    0.294118,   0.301961,   0.305882,   0.313726,   0.32549,    0.337255,   0.345098,   0.352941,   0.368627,   0.380392,   0.396078,   0.403922,   0.415686,   0.431373,   0.443137,   0.454902,   0.478431,   0.494118,  0.501961,  0.521569,  0.54902,   0.588235,  0.623529,  0.682353,  0.760784,  0.815686},
/* 21 */            {0.0},
/* 22 */            {0.0},
/* 23 */            {0.0,        0.0166667,  0.02,       0.0227273,   0.030303,   0.03125,    0.0333333,  0.0454545,  0.047619,   0.0588235,  0.0606061,  0.0666667,  0.0714286,  0.0909091,  0.1,        0.111111,   0.117647,   0.125,      0.142857,   0.181818,   0.222222,   0.25,       0.3,       0.333333,  0.5,       0.666667,  1.0,       1.5},
/* 24 */            {0.050778,   0.0602437,  0.0819724,  0.115745,    0.174157,   0.181041,   0.185779,   0.192457,   0.201191,   0.208477,   0.215301,   0.221551,   0.228534,   0.237347,   0.246315,   0.30681,    0.346379,   0.456995,   0.57209,    0.83256,    0.930972,   0.937682,   0.937938,  0.938279,  0.938714,  0.939347,  0.940426,  0.942513,  0.946532,  0.953957,  0.96522},
/* 25 */            {0.0,        0.5,        0.666667},
/* 26 */            {0.0},
/* 27 */            {0.0},
/* 28 */            {0.0},
/* 29 */            {},
/* 30 */            {},
/* 31 */            {0.0,        0.00990099, 0.0196078,  0.0291262,   0.0384615,  0.0476191,  0.0566038,  0.0654206,  0.0740741,  0.0825688,  0.0909091,  0.107143,   0.122807,   0.130435,   0.145299,   0.152542,   0.173554,   0.193548,   0.212598,   0.242424,   0.275362,   0.305556,   0.342105,  0.378882,  0.425287,  0.479167,  0.53917,   0.615385,  0.671053,  0.742931,  0.844237},
/* 32 */            {0.0},
/* 33 */            {0.0,        0.004854,   0.009345,   0.015037,    0.02649,    0.03409,    0.044198,   0.057142,   0.075235,   0.092896,   0.111702,   0.138297,   0.165938,   0.197132,   0.215447,   0.249266,   0.25,       0.272727,   0.330769,   0.333333,   0.398169,   0.404761,   0.434782,  0.491329,  0.5,       0.584112,  0.648648,  0.666666,  0.733333,  0.807947,  0.888888},
/* 34 */            {0.0},
/* 35 */            {-0.357361,  -0.204348,  -0.101202,  -1.63667E-4, 0.0,        0.0778547,  0.134873,   0.196173,   0.247038,   0.29992,    0.346679,   0.392045,   0.434363,   0.472441,   0.518501,   0.554864,   0.597666,   0.636878,   0.65629,    0.738653,   0.772688,   0.774959,   0.78228,   0.814661,  0.885729,  0.93951,   0.950154,  0.965436,  0.985321,  0.994094,  1.00879},
/* 36 */            {0.0},
/* 37 */            {0.0,        1.68464E-4, 3.36927E-4, 5.05391E-4,  6.73854E-4, 8.42318E-4, 0.00101078, 0.00117925, 0.00151617, 0.00202156, 0.00235849, 0.00286388, 0.00404313, 0.00471698, 0.0055593,  0.00758086, 0.00892857, 0.0102763,  0.0121294,  0.0141509,  0.0166779,  0.0197102,  0.0229111, 0.0281334, 0.0338612, 0.039252,  0.0475067, 0.0574461, 0.0707547, 0.0887382, 0.134444},
/* 38 */            {0.0,        0.0485252,  0.15683,    0.606299,    0.651447,   0.651689,   0.651811,   0.652053,   0.652416,   0.652898,   0.6535,     0.653979,   0.654338,   0.655647,   0.656711,   0.657417,   0.658236,   0.661476,   0.663639,   0.666667,   0.669093,   0.673948,   0.682338,  0.693346,  0.72429,   0.746128,  0.782277,  0.824838,  0.877795,  0.914879,  0.944831},
/* 39 */            {0.0,        6.06061E-5, 1.42424E-4, 2.47619E-4,  3.17661E-4, 3.82796E-4, 5.41558E-4, 7.05195E-4, 8.75822E-4, 0.00108394, 0.00135281, 0.0016479,  0.00194805, 0.00233952, 0.00273569, 0.00325677, 0.003848,   0.00452341, 0.00532332, 0.00627706, 0.00741142, 0.00884064, 0.010582,  0.0123647, 0.0142519, 0.0167411, 0.0196293, 0.0231264, 0.0278319, 0.0339218, 0.0439715},
/* 40 */            {},
/* 41 */            {},
/* 42 */            {},
/* 43 */            {},
/* 44 */            {},
/* 45 */            {},
/* 46 */            {0.00911577, 0.0532194,  0.0840336,  0.112097,    0.135452,   0.157211,   0.178571,   0.19933,    0.218785,   0.237414,   0.25511,    0.273821,   0.294104,   0.312135,   0.320482,   0.329477,   0.347804,   0.366323,   0.388651,   0.408512,   0.429201,   0.451329,   0.46886,   0.489662,  0.511167,  0.533333,  0.56074,   0.585209,  0.615375,  0.652835,  0.716116},
/* 47 */            {0.0, 0.223209, 0.247053, 0.263378, 0.285018, 0.285714, 0.364835, 0.412443, 0.44435, 0.444444, 0.544606, 0.545455, 0.614718, 0.615385, 0.657497, 0.677752, 0.707054, 0.736842, 0.761905, 0.796921, 0.814815, 0.857242, 0.878007, 0.897236, 0.911612, 0.928389, 0.941387, 0.953294, 0.964422, 0.975118, 0.982886},
/* 48 */            {0.0, 0.02598, 0.0654359, 0.0965742, 0.124069, 0.146987, 0.168067, 0.188639, 0.206186, 0.227243, 0.244268, 0.266886, 0.289849, 0.312471, 0.334834, 0.356995, 0.378357, 0.399506, 0.420104, 0.441082, 0.462398, 0.483871, 0.504507, 0.527554, 0.553398, 0.58011, 0.608649, 0.643941, 0.68937, 0.751795, 0.844118},
/* 49 */            {0.0},
/* 50?*/            {}
    };


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

BinarizationConfig BinarizationConfig::fromJson(const json& params) {
    BinarizationConfig opts;
    opts.bordersCount_ = params["borders_count"];
    return opts;
}

std::vector<float> buildBorders(const BinarizationConfig& config, Vec* vals) {
    std::vector<float> borders;
    if (vals->dim()) {
        auto sortedFeature = VecFactory::toDevice(VecTools::sort(*vals), ComputeDevice(ComputeDeviceType::Cpu));
        auto data = sortedFeature.arrayRef();
        for (uint32_t i = 1; i < data.size(); ++i) {
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
//        std::vector<float> featureBorders = jmllFBorders[fIndex];
        if (!featureBorders.empty()) {
            std::cout << "fId=" << nzFeatureId << ", borders: ";
            for (auto border : featureBorders) {
                std::cout << border << ", ";
            }
            std::cout << std::endl;
            borders.push_back(featureBorders);
            const auto binCount = borders.back().size();
            features.push_back(Feature(nzFeatureId, binCount, fIndex));
            for (uint32_t bin = 0; bin < binCount; ++bin) {
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

