#pragma once
#include <torch/torch.h>
#include <core/object.h>
#include <core/matrix.h>
#include <core/cache.h>

class DataSet : public Object, public CacheHolder<DataSet> {
public:
    explicit DataSet(Mx data, Vec target)
    : data_(data)
    , target_(target){
        assert(target.dim() == samplesCount());
    }

    int64_t featuresCount() const {
        return data_.xdim();
    }

    int64_t samplesCount() const {
        return data_.ydim();
    }

    void copyColumn(int fIndex, Vec* col) const {
        assert(col->dim() == samplesCount());
        assert(col->isContiguous());
        VecRef<float> writeDst = col->arrayRef();
        data_.iterateOverColumn(fIndex, [&](int64_t lineIdx, float val) {
            writeDst[lineIdx] = val;
        });
    }


    template <class Visitor>
    void visitColumn(int fIndex, Visitor&& visitor) const {
        data_.iterateOverColumn(fIndex, visitor);
    }

    Vec sample(int64_t line) const {
        return data_.row(line);
    }

    Vec target() const {
        return target_;
    }

    const float* samples() const {
        return data_.arrayRef().data();
    }
    const float* labels() const {
        return target_.arrayRef().data();
    }
private:
    Mx data_;
    Vec target_;
};
