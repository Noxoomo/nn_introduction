#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <cassert>
#include <util/array_ref.h>


template <class AdditiveStat,
          class I,
          int64_t BundleSize = 4,
          int64_t N = 8>
void buildHistograms(
    ConstVecRef<AdditiveStat> statistics,
    ConstVecRef<I> binLoadIndices,
    ConstVecRef<int32_t> binOffsets,
    ConstVecRef<uint8_t> data,
    VecRef<AdditiveStat> dst) {
    std::array<int32_t, BundleSize * N> localBins;
    const auto size = static_cast<const int64_t>(binLoadIndices.size());

    for (int64_t i = 0; i < (size / N) * N; i += N) {
        for (int64_t k = 0; k < N; ++k) {
            const int64_t loadIdx = binLoadIndices[i + k];
            for (int64_t b = 0; b < BundleSize; ++b) {
                localBins[b * N + k] = data[loadIdx * BundleSize + b];
            }
        }

        for (int64_t b = 0; b < BundleSize; ++b) {
            for (int64_t k = 0; k < N; ++k) {
                dst[binOffsets[b] + localBins[b * N + k]] +=  statistics[i + k];
            }
        }
    }

    for (int64_t i = ((size / N) * N); i < size; ++i) {
        for (int64_t b = 0; b < BundleSize; ++b) {
            const int64_t loadIdx = binLoadIndices[i];
            dst[binOffsets[b] + data[loadIdx * BundleSize + b]] += statistics[i];
        }
    }
}


template <class AdditiveStat,
        class I,
        int64_t N = 4>
void buildHistograms(
    int32_t bundleSize,
    ConstVecRef<AdditiveStat> statistics,
    ConstVecRef<I> binLoadIndices,
    ConstVecRef<int32_t> binOffsets,
    ConstVecRef<uint8_t> data,
    VecRef<AdditiveStat> dst) {

    assert(bundleSize <= 64);
    #define DISPATCH(sz)\
    buildHistograms<AdditiveStat, I, sz, N>(statistics, binLoadIndices, binOffsets, data, dst);

    #define DISPATCH_GE32(sz)\
    buildHistograms<AdditiveStat, I, sz + 32, N>(statistics, binLoadIndices, binOffsets, data, dst);


    if (bundleSize > 32) {
        bundleSize -= 32;
        if (bundleSize == 32) {
            DISPATCH_GE32(32)
        } else if (bundleSize == 31) {
            DISPATCH_GE32(31);
        } else if (bundleSize == 30) {
            DISPATCH_GE32(30);
        } else if (bundleSize == 29) {
            DISPATCH_GE32(29);
        } else if (bundleSize == 28) {
            DISPATCH_GE32(28);
        } else if (bundleSize == 27) {
            DISPATCH_GE32(27);
        } else if (bundleSize == 26) {
            DISPATCH_GE32(26);
        } else if (bundleSize == 25) {
            DISPATCH_GE32(25);
        } else if (bundleSize == 24) {
            DISPATCH_GE32(24);
        } else if (bundleSize == 23) {
            DISPATCH_GE32(23);
        } else if (bundleSize == 22) {
            DISPATCH_GE32(22);
        } else if (bundleSize == 21) {
            DISPATCH_GE32(21);
        } else if (bundleSize == 20) {
            DISPATCH_GE32(20);
        } else if (bundleSize == 19) {
            DISPATCH_GE32(19);
        } else if (bundleSize == 18) {
            DISPATCH_GE32(18);
        } else if (bundleSize == 17) {
            DISPATCH_GE32(17);
        } else if (bundleSize == 16) {
            DISPATCH_GE32(16)
        } else if (bundleSize == 15) {
            DISPATCH_GE32(15);
        } else if (bundleSize == 14) {
            DISPATCH_GE32(14);
        } else if (bundleSize == 13) {
            DISPATCH_GE32(13);
        } else if (bundleSize == 12) {
            DISPATCH_GE32(12);
        } else if (bundleSize == 11) {
            DISPATCH_GE32(11);
        } else if (bundleSize == 10) {
            DISPATCH_GE32(10);
        } else if (bundleSize == 9) {
            DISPATCH_GE32(9);
        } else if (bundleSize == 8) {
            DISPATCH_GE32(8);
        } else if (bundleSize == 7) {
            DISPATCH_GE32(7);
        } else if (bundleSize == 6) {
            DISPATCH_GE32(6);
        } else if (bundleSize == 5) {
            DISPATCH_GE32(5);
        } else if (bundleSize == 4) {
            DISPATCH_GE32(4);
        } else if (bundleSize == 3) {
            DISPATCH_GE32(3);
        } else if (bundleSize == 2) {
            DISPATCH_GE32(2);
        } else if (bundleSize == 1) {
            DISPATCH_GE32(1);
        } else {
            assert(false);
        }
    } else {
        if (bundleSize == 32) {
            DISPATCH(32)
        } else if (bundleSize == 31) {
            DISPATCH(31);
        } else if (bundleSize == 30) {
            DISPATCH(30);
        } else if (bundleSize == 29) {
            DISPATCH(29);
        } else if (bundleSize == 28) {
            DISPATCH(28);
        } else if (bundleSize == 27) {
            DISPATCH(27);
        } else if (bundleSize == 26) {
            DISPATCH(26);
        } else if (bundleSize == 25) {
            DISPATCH(25);
        } else if (bundleSize == 24) {
            DISPATCH(24);
        } else if (bundleSize == 23) {
            DISPATCH(23);
        } else if (bundleSize == 22) {
            DISPATCH(22);
        } else if (bundleSize == 21) {
            DISPATCH(21);
        } else if (bundleSize == 20) {
            DISPATCH(20);
        } else if (bundleSize == 19) {
            DISPATCH(19);
        } else if (bundleSize == 18) {
            DISPATCH(18);
        } else if (bundleSize == 17) {
            DISPATCH(17);
        } else if (bundleSize == 16) {
            DISPATCH(16)
        } else if (bundleSize == 15) {
            DISPATCH(15);
        } else if (bundleSize == 14) {
            DISPATCH(14);
        } else if (bundleSize == 13) {
            DISPATCH(13);
        } else if (bundleSize == 12) {
            DISPATCH(12);
        } else if (bundleSize == 11) {
            DISPATCH(11);
        } else if (bundleSize == 10) {
            DISPATCH(10);
        } else if (bundleSize == 9) {
            DISPATCH(9);
        } else if (bundleSize == 8) {
            DISPATCH(8);
        } else if (bundleSize == 7) {
            DISPATCH(7);
        } else if (bundleSize == 6) {
            DISPATCH(6);
        } else if (bundleSize == 5) {
            DISPATCH(5);
        } else if (bundleSize == 4) {
            DISPATCH(4);
        } else if (bundleSize == 3) {
            DISPATCH(3);
        } else if (bundleSize == 2) {
            DISPATCH(2);
        } else if (bundleSize == 1) {
            DISPATCH(1);
        } else {
            assert(false);
        }
    }

    #undef DISPATCH
    #undef DISPATCH_GE32
}
