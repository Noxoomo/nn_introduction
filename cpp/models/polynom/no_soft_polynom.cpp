#include <iostream>

void PolynomForward(
    const float lambda,
    const float* features,
    int fCount,
    int batchSize,
    const int* splits,
    const float* conditions,
    const int* polynomOffsets,
    const float* values,
    int polynomCount,
    int outDim,
    float* tempProbs,
    float* output
) {
    std::cout << "No cuda support " << std::endl;
    throw std::exception();
}


void PolynomBackward(const float* features,
                     int featuresCount,
                     int batchSize,
                     const float* outDer,
                     int outputDim,
                     const float* leafSum,
                     int* polynomOffset,
                     int* featureIds,
                     float* conditions,
                     int polynomCount,
                     float* out) {
    std::cout << "No cuda support " << std::endl;
    throw std::exception();
}