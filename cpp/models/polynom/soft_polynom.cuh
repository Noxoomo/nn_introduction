#pragma once

#include <cuda_runtime.h>

void PolynomBackward(const float* features,
                     int featuresCount,
                     int batchSize,
                     const float* outDer,
                     int outputDim,
                     const float* leafSum,
                     int* polynomDepths,
                     int* polynomOffset,
                     int* featureIds,
                     float* conditions,
                     int polynomCount,
                     float* out,
                     cudaStream_t stream);