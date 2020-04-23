#pragma once

#include <core/vec.h>
#include <core/buffer.h>

namespace VecTools {

    Vec copy(const Vec& other);

    Vec copyTo(const Vec& from, Vec to);

    Vec subtract(Vec x, const Vec& y);

    Vec pow(Scalar p, const Vec& from, Vec to);

    Vec pow(Scalar p, Vec x);

    Vec sign(const Vec& x, Vec to);

    Vec abs(Vec to);

    Vec expCopy(Vec x);

    Vec log(Vec x);

    Vec absCopy(const Vec& source);

    Vec mul(const Vec& x, Vec y);

    Vec mul(Scalar alpha, Vec x);

    Vec sigmoidCopy(const Vec& point);


    Vec gather(const Vec& from, const Buffer<int32_t>& map, Vec to);
}
