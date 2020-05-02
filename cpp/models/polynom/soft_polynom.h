#include <iostream>
#include <exception>

void ExpProbPolynomForward(
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
    );

void ExpProbPolynomBackward(int batchSize,
                     float lambda,
                     const float* features,
                     int featuresCount,
                     const float* outDer,
                     int outputDim,
                     const int* featureIds,
                     const float* conditions,
                     const float* values,
                     const int* polynomOffsets,
                     int polynomCount,
                     float* featuresDer);

void SigmoidProbPolynomForward(
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
);

void SigmoidProbPolynomBackward(int batchSize,
                     float lambda,
                     const float* features,
                     int featuresCount,
                     const float* outDer,
                     int outputDim,
                     const int* featureIds,
                     const float* conditions,
                     const float* values,
                     const int* polynomOffsets,
                     int polynomCount,
                     float* featuresDer);

void LinearPolynomForward(
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
        float* output,
        const int* origFId
);

void LinearPolynomBackward(int batchSize,
                           float lambda,
                           const float* features,
                           int featuresCount,
                           const float* outDer,
                           int outputDim,
                           const int* featureIds,
                           const float* conditions,
                           const float* values,
                           const int* polynomOffsets,
                           int polynomCount,
                           float* featuresDer,
                           const int* origFId);
