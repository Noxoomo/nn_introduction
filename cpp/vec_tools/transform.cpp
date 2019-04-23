#include "transform.h"
#include <core/vec_factory.h>
#include <core/buffer.h>

namespace VecTools {

    Vec copyTo(const Vec& from, Vec to) {
        to.data().copy_(from);
        return to;
    }

    Vec copy(const Vec& other) {
        return VecFactory::clone(other);
    }

    Vec subtract(Vec x, const Vec& y) {
        x -= y;
        return x;
    }

    Vec pow(Scalar p, const Vec& from, Vec to) {
        assert(from.dim() == to.dim());
        at::pow_out(to, from, p);
        return to;
    }

    Vec mul(const Vec& x, Vec y) {
        assert(x.dim() == y.dim());
        y *= x;
        return y;
    }

    Vec mul(Scalar alpha, Vec x) {
        x.data() *= (float) alpha;
        return x;
    }

    Vec pow(Scalar p, Vec x) {
        x.data().pow(p);
        return x;
    }

    Vec sign(const Vec& x, Vec to) {
        assert(x.dim() == to.dim());
        at::sign_out(to, x);
        return to;
    }

    Vec abs(Vec x) {
        at::abs_(x.data());
        return x;
    }

    Vec absCopy(const Vec& x) {
        return abs(copy(x));
    }

    Vec expCopy(Vec x) {
        return Vec(x.data().exp());
    }
    Vec log(Vec x) {
        x.data().log_();
        return x;
    }

    template <class T, class I, class TC>
    inline void gatherCpuImpl(ConstVecRef<T> from, ConstVecRef<I> map, VecRef<TC> to) {
        for (int64_t i = 0; i < map.size(); ++i) {
            to[i] = from[map[i]];
        }
    }


    template <class T, class I, class TC>
    inline void scatterCpuImpl(ConstVecRef<T> from, ConstVecRef<I> map, VecRef<TC> to) {
        for (int64_t i = 0; i < map.size(); ++i) {
            to[map[i]] = from[i];
        }
    }


    Vec gather(const Vec& from, const Buffer<int32_t>& map, Vec to) {
        VERIFY(from.isContiguous() && from.isCpu(), "error: not implemented on gpu yet");
        gatherCpuImpl(from.arrayRef(), map.arrayRef(), to.arrayRef());
        return to;
    }


}
