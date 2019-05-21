#include "polynom.h"
#include <array>
#include <cmath>
#include <map>
#include <util/exception.h>
#include <util/parallel_executor.h>
#include <iostream>

struct PathBit {
    int Bits = 0;
    int Sign = 1;
};

inline std::vector<PathBit> LeafToPolynoms(const int path, int maxDepth) {
    std::vector<PathBit> pathBits = {{}};
    for (int depth = 0; depth <  maxDepth; ++depth) {
        const int mask = 1 << depth;
        const bool isOne = path & mask;

        if (isOne) {
            for (auto& bit : pathBits) {
                bit.Bits |= 1 << depth;
            }
        } else {
            uint64_t currentPaths = pathBits.size();
            for (uint64_t i = 0; i < currentPaths; ++i) {
                auto bit = pathBits[i];
                bit.Bits |= 1 << depth;
                bit.Sign *= -1;
                pathBits.push_back(bit);
            }
        }
    }
    return pathBits;
}

template <class Vec>
inline void SortUnique(Vec& vec) {
    std::sort(vec.begin(), vec.end());
    vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
}

void PolynomBuilder::AddTree(const TSymmetricTree& tree)  {
    const int maxDepth = static_cast<int>(tree.Conditions.size());
    if (maxDepth == 0) {
        return;
    }
    const int leaves = 1 << maxDepth;

    std::vector<double> weights(leaves);
    std::vector<double> values(leaves * tree.OutputDim());

    const int outputDim = tree.OutputDim();
    for (int path = 0; path < leaves; ++path) {
        auto polynoms = LeafToPolynoms(path, maxDepth);

        for (const auto& polynom : polynoms) {
            int idx = polynom.Bits;
            for (int dim = 0; dim < outputDim; ++dim) {
                values[idx * outputDim + dim] += polynom.Sign * tree.Leaves[path * outputDim + dim];
            }
        }
    }
    for (int path = 0; path < leaves; ++path) {
        for (int i = 0; i < leaves; ++i) {
            if ((i & path) == path) {
                weights[path] += tree.Weights[i];
            }
        }
    }


    for (int i = 0; i < leaves; ++i) {
        // co c1 (1 - c2)
        //if 0 points in c0 c1 c2 => -c0c1c2 with weight 0, but valuee will not be zero
//        if (weights[i] == 0) {
//            continue;
//        }
        PolynomStructure polynomStructure;
        std::map<int, float> polynom;

        for (int depth = 0; depth < maxDepth; ++depth) {
            int mask = 1 << depth;
            if (i & mask) {
                BinarySplit split;
                split.Feature = tree.Features[depth];
                split.Condition = tree.Conditions[depth];
                if (polynom.count(split.Feature)) {
                    polynom[split.Feature] = std::max<float>(split.Condition, polynom[split.Feature]);
                } else {
                    polynom[split.Feature] = split.Condition;
                }
            }
        }
        for (const auto& [split, condition] : polynom) {
            polynomStructure.Splits.push_back({split, condition});
        }
        SortUnique(polynomStructure.Splits);
        auto& dst = EnsemblePolynoms[polynomStructure];
        if (dst.Weight < 0) {
            dst.Weight = weights[i];
        } else {
            VERIFY(dst.Weight == weights[i], "Weight for fixed path should be equal for all polynoms");
        }
        dst.Value.resize(tree.OutputDim());
        for (uint32_t dim = 0; dim < tree.OutputDim(); ++dim) {
            dst.Value[dim] += values[i * tree.OutputDim() + dim];
        }
    }

}


void Monom::Forward(double lambda, ConstVecRef<float> features, VecRef<float> dst) const {
//    bool allTrue = true;
//    for (const auto& split : Structure_.Splits) {
//
//        if (features[split.Feature] <= split.Condition) {
//            allTrue = false;
//            break;
//        }
//    }
//    if (allTrue) {
//        for (int dim = 0; dim < dst.size(); ++dim) {
//            dst[dim] +=  Values_[dim];
//        }
//    }
  double trueLogProb = 0;
  for (const auto& split : Structure_.Splits) {
    const double val = -lambda * (features[split.Feature] - split.Condition);
//    log(1.0 / (1.0 + exp(-val))) = -log(1.0 + exp(-val));

    const double expVal = exp(val);
    if (std::isfinite(expVal)) {
      trueLogProb -= log(1.0 + expVal);
    } else {
      trueLogProb -= val;
    }
  }
  const double p = exp(trueLogProb);
  for (int dim = 0; dim < dst.size(); ++dim) {
    dst[dim] += p * Values_[dim];
  }
}

void Monom::Backward(double lambda,
                     ConstVecRef<float> features,
                     ConstVecRef<float> outputsDer,
                     VecRef<float> featuresDer) const {

  std::array<double, 8> logProbs;
  logProbs.fill(0.0f);//Structure_.Splits.size());

  double totalLogProb = 0;

  for (int i = 0; i < Structure_.Splits.size(); ++i) {
    const auto& split = Structure_.Splits[i];
    const double val = -lambda * (features[split.Feature] - split.Condition);
//    log(1.0 / (1.0 + exp(-val))) = -log(1.0 + exp(-val));

    const double expVal = exp(val);
    if (std::isfinite(expVal)) {
      logProbs[i] -= log(1.0 + expVal);
    } else {
      logProbs[i] -= val;
    }
    totalLogProb += logProbs[i];
  }
  const double p = exp(totalLogProb);

  double tmp = 0;
  for (size_t dim = 0; dim < Values_.size(); ++dim) {
      tmp += Values_[dim] * outputsDer[dim];
  }

  for (int i = 0; i < Structure_.Splits.size(); ++i) {
    const auto& split = Structure_.Splits[i];
    const double featureProb = exp(logProbs[i]);
    const double monomDer = p * (1.0 - featureProb);
    featuresDer[split.Feature] += monomDer * tmp;
  }
}


void Polynom::Forward(ConstVecRef<float> features,
                      VecRef<float> dst) const {
    const int threadCount = GlobalThreadPool().numThreads();

    std::vector<std::vector<float>> partResults(threadCount, std::vector<float>(dst.size()));

    parallelFor(0, threadCount, [&](int i) {
        const int elemsPerBlock = (Ensemble_.size() + threadCount - 1) / threadCount;
        const int blockStart = i  * elemsPerBlock;
        const int blockEnd = std::min<int>(blockStart + elemsPerBlock, Ensemble_.size());
        for (int j = blockStart; j < blockEnd; ++j) {
            Ensemble_[j].Forward(Lambda_, features, partResults[i]);
        }
    });

    for (int i = 0; i < threadCount; ++i) {
        for (int j = 0; j < dst.size(); ++j) {
            dst[j] += partResults[i][j];
        }
    }
}

void Polynom::Backward(ConstVecRef<float> features,
                       ConstVecRef<float> outputsDer,
                       VecRef<float> featuresDer) const {
    for (const auto& monom : Ensemble_) {
        monom.Backward(Lambda_, features, outputsDer, featuresDer);
    }
}

void Polynom::PrintHistogram() {
    for (int dim = 0; dim < OutDim(); ++dim) {
        double min = Ensemble_.back().Values_[dim];
        double max = min;
        for (const auto& monom : Ensemble_) {
            min = std::min<double>(monom.Values_[dim], min);
            max = std::max<double>(monom.Values_[dim], max);
        }
        const int binCount = 32;

        std::vector<int> bins(binCount);
        double total = 0;
        for (const auto& monom : Ensemble_) {
            double v = monom.Values_[dim];
            int bin = (v - min) * binCount / (max - min);
            bin = std::max<int>(bin, 0);
            bin = std::min<int>(bin, binCount - 1);
            ++bins[bin];
            ++total;
        }
        std::cout << "dim=" << dim << ", min=" << min <<", max=" << max << std::endl;
        for (const auto& binCount : bins) {
            std::cout << binCount * 1.0 / total << " ";
        }
        std::cout << std::endl;
    }
}
