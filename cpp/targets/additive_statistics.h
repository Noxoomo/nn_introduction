#pragma once

#include <vector>

#include <util/array_ref.h>

namespace {
    using AdditiveStatisticsOpEmptyParams = struct {};

    struct AdditiveStatisticsDefaultTypeTraits {
        using ImplSampleType = const float*;
        using ImplTargetType = float;
        using ImplWeightType = float;
    };
}

template <typename Impl, typename ImplTypeTraits = AdditiveStatisticsDefaultTypeTraits, typename OpParams = AdditiveStatisticsOpEmptyParams>
class AdditiveStatistics {
public:
    using SampleType = typename ImplTypeTraits::ImplSampleType;
    using TargetType = typename ImplTypeTraits::ImplTargetType;
    using WeightType = typename ImplTypeTraits::ImplWeightType;

    Impl& append(const AdditiveStatistics& other, const OpParams& opParams) {
        return static_cast<Impl*>(this)->appendImpl(static_cast<const Impl&>(other), opParams);
    }

    Impl& remove(const AdditiveStatistics& other, const OpParams& opParams) {
        return static_cast<Impl*>(this)->removeImpl(static_cast<const Impl&>(other), opParams);
    }

    Impl& append(SampleType x, TargetType y, WeightType weight, const OpParams& opParams) {
        return static_cast<Impl*>(this)->appendImpl(x, y, weight, opParams);
    }

    Impl& remove(SampleType x, TargetType y, WeightType weight, const OpParams& opParams) {
        return static_cast<Impl*>(this)->removeImpl(x, y, weight, opParams);
    }

    Impl& operator+=(const AdditiveStatistics& other) {
        OpParams defaultParams;
        return append(other, defaultParams);
    }

    Impl& operator-=(const AdditiveStatistics& other) {
        OpParams defaultParams;
        return remove(other, defaultParams);
    }

    Impl operator+(const AdditiveStatistics& other) {
        Impl tmp(static_cast<Impl&>(*this));
        tmp += other;
        return tmp;
    }

    Impl operator-(const AdditiveStatistics& other) {
        Impl tmp(static_cast<Impl&>(*this));
        tmp -= other;
        return tmp;
    }
};
