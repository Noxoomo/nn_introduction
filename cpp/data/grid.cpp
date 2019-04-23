#include "grid.h"

void Grid::binarize(ConstVecRef<float> row, VecRef<uint8_t> dst) const {
    for (int64_t f = 0; f < features_.size(); ++f) {
        dst[f] = computeBin(row[origFeatureIndex(f)], borders_[f]);
    }
}


void Grid::binarize(const Vec& x, Buffer<uint8_t>& to) const {
    assert(x.device().deviceType() == ComputeDeviceType::Cpu);
    assert(to.device().deviceType() == ComputeDeviceType::Cpu);

    const auto values = x.arrayRef();
    const auto dst = to.arrayRef();
    binarize(values, dst);
}
