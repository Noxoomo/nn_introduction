#include "grid.h"

#include <util/io.h>

void Grid::binarize(ConstVecRef<float> row, VecRef<uint8_t> dst) const {
    for (uint64_t f = 0; f < features_.size(); ++f) {
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

void Grid::serialize(std::ostream& out) const {
    out.write("g{", 2);

    int fCount = fCount_;
    out.write((char*)&fCount, sizeof(fCount));

    int nzFCount = nzFeaturesCount();
    writeEnclosed(out, &nzFCount, sizeof(nzFCount), "s");

    for (int fId = 0; fId < (int)nzFeaturesCount(); ++fId) {
        int origFId = origFeatureIndex(fId);
        out.write((char*)&origFId, sizeof(origFId));
        ConstVecRef<float> fborders = borders(fId);
        int size = fborders.size();
        writeEnclosed(out, &size, sizeof(size), "s");
        for (float border : fborders) {
            out.write((char*)&border, sizeof(border));
        }
    }

    out.write("}", 1);
}
