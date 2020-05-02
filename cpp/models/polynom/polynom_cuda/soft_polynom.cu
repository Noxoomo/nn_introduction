#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#if __CUDA_ARCH__ < 350
template <typename T>
__forceinline__ __device__ T __ldg(const T* data) {
    return data[0];
}
#endif

__forceinline__ __device__ float Sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ float SigmoidDer(float x) {
    const float p = 1.0f / (1.0f + expf(-x));
    return p * (1.0f - p);
}

// ExpProb Polynom {{{

__global__ void ExpProbPolynomProbsImpl(
    const float* features,
    int batchSize,
    const int* splits,
    const float* conditions,
    const int* polynomOffsets,
    int polynomCount,
    float lambda,
    float* probs) {
    if (threadIdx.x < batchSize) {
        int polynomId = blockIdx.x;

        features +=  threadIdx.x;
        probs += threadIdx.x;

        while (polynomId < polynomCount) {
            int offset = polynomOffsets[polynomId];
            int nextOffset = polynomOffsets[polynomId + 1];
            const int depth = nextOffset - offset;

            float logProb = 0;
            bool zeroProb = false;
            for (int i = 0; i < depth; ++i) {
                if (zeroProb) {
                    continue;
                }

                const int f = __ldg(splits + offset + i);
                const float c = __ldg(conditions + offset + i);
                const float x = __ldg(features + f * batchSize);

                const float val = -lambda * x;
                const float expVal = 1.0f - expf(val);

                if (isfinite(log(expVal))) {
                    logProb += log(expVal);
                } else {
                    zeroProb = true;
                }
            }

            float prob = 0.0f;
            if (!zeroProb) {
                prob = expf(logProb);
            }

            probs[polynomId * batchSize] = prob;
            polynomId += gridDim.x;
        }
    }
}

//batch size should be equal to BlockSize
//we need to reduce polynoms for each output dim
__global__ void ExpProbPolynomForwardImpl(
    const float* probs,
    int batchSize,
    const float* values,
    int polynomCount,
    int outputDim,
    float* out) {

    //out: batch_elem0 dim0, dim1, dimk batch_elem1 dim0 dim1 dimk
    //so threads
    int polynomId = blockIdx.x;
    const int dimId = blockIdx.y;

    int tid = threadIdx.x;
    if (tid >= batchSize) {
        return;
    }

    float sum = 0;
    probs += threadIdx.x;
    values += dimId;

    while (polynomId < polynomCount) {
        const float polynomProb = __ldg(probs + polynomId * batchSize);
        const float out = __ldg(values + polynomId * outputDim);
        sum += polynomProb * out;
        polynomId += gridDim.x;
    }

    atomicAdd(out + dimId * batchSize + threadIdx.x, sum);
}

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
) {
    const int blockSize = batchSize;
    const int numBlocks = min(polynomCount, 1000);
    assert(batchSize < 2048);
    assert(numBlocks);

    ExpProbPolynomProbsImpl << < numBlocks, blockSize >>> (features, batchSize, splits, conditions, polynomOffsets, polynomCount, lambda, tempProbs);

    dim3 forwardBlocks;
    forwardBlocks.z = 1;
    forwardBlocks.y = outDim;
    forwardBlocks.x = min(polynomCount, 512);
    ExpProbPolynomForwardImpl << < forwardBlocks, batchSize >> > (tempProbs, batchSize, values, polynomCount, outDim, output);
}

/*
 * Here layout is not the same as in forward pass
 * BlockSize = 256, MaxDepth = 8, K = 24
 * should give 50% occupancy, this should be enough
 */
template <int MaxDepth, int BlockSize, int K>
__global__ void ExpProbPolynomBackwardImpl(float lambda,
                                    const float* features,
                                    int featuresCount,
                                    const float* outDer,
                                    int outputDim,
                                    const int* featureIds,
                                    const float* conditions,
                                    const float* values,
                                    const int* polynomOffsets,
                                    int polynomCount,
                                    float* featuresDer) {
    const int sampleId = blockIdx.y;

    features += sampleId * featuresCount;
    featuresDer += sampleId * featuresCount;
    outDer += sampleId * outputDim;


    //out: batch_elem0 dim0, dim1, dimk batch_elem1 dim0 dim1 dimk
    //so threads

    __shared__ float localFeaturesDer[BlockSize * K];

    for (int i = threadIdx.x; i < BlockSize * K; i += BlockSize) {
        localFeaturesDer[i] = 0;
    }
    __syncthreads();


    const int alignedFeaturesCount = ((featuresCount + BlockSize - 1) / BlockSize) * BlockSize;

    const int memoryBlocks = (BlockSize * K) / alignedFeaturesCount;
    const int memoryBlockId = threadIdx.x % memoryBlocks;


    int polynomId = blockIdx.x * blockDim.x + threadIdx.x;

    while (polynomId < polynomCount) {
        const int offset = polynomOffsets[polynomId];
        const int nextOffset = polynomOffsets[polynomId + 1];
        const int depth = nextOffset - offset;

        if (depth != 0) {

            float logProbs[MaxDepth];
            float vals[MaxDepth];
            short fids[MaxDepth];
            float totalLogProb = 0;

            bool zeroProb = false;

            #pragma unroll
            for (int i = 0; i < MaxDepth; ++i) {
                if (i < depth) {
                    const int f = __ldg(featureIds + i + offset);
                    fids[i] = f;
                    const float c = __ldg(conditions + i + offset);
                    const float x = __ldg(features + f);

                    vals[i] = -lambda * x;
                    const float expVal = 1.0f - exp(vals[i]);
                    logProbs[i] = log(expVal);
                    if (isfinite(logProbs[i])) {
                        totalLogProb += logProbs[i];
                    } else {
                        zeroProb = true;
                    }
                }
            }

            //featureDerivative is outputDer * total value before monom * monom derivative
            float derMultiplier = 0;
            #pragma unroll 10
            for (int dim = 0; dim < outputDim; ++dim) {
                derMultiplier += __ldg(values + polynomId * outputDim + dim) * __ldg(outDer + dim);
            }

            #pragma unroll
            for (int i = 0; i < MaxDepth; ++i) {
                if (i < depth) {
                    const int f = fids[i];

                    // XXX for zero feature it actually shouldn't be zero, but it's not propagated through relu anyways.
                    float featureDer = 0.0f;
                    if (!zeroProb) {
                        // (1 - e^{-lambda * x})' = lambda * e^{-lambda * x}
                        //
                        // dp / dx_i = p / (1 - e^{-l * x_i}) * (l * e^{-l * x_i})
                        // ln (p / (1 - e^{-l * x}) * (l * e^{-l * x})) = ln(p) - ln(1 - e^{-x}) + ln(l) + (-l * x))
                        const float monomDer = exp(totalLogProb - logProbs[i] + log(lambda) + vals[i]);
                        featureDer = monomDer * derMultiplier;
                    }

                    //atomics in shared memory, pretty fast on pascal+ hardware
                    atomicAdd(localFeaturesDer + memoryBlocks * f + memoryBlockId, featureDer);
                }
            }
        }
        polynomId += gridDim.x * blockDim.x;
    }


    __syncthreads();

    //outputDim = 1024 => memoryBlocks = 6
    for (int f = threadIdx.x; f < featuresCount; f += BlockSize) {
        float der = 0;

        #pragma unroll
        for (int k = 0; k < memoryBlocks; ++k) {
            der += localFeaturesDer[f * memoryBlocks + k];
        }
        atomicAdd(featuresDer + f,  der);
    }
}

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
                                float* featuresDer) {

    const int blockSize = 256;
    dim3 numBlocks;
    numBlocks.z = 1;
    numBlocks.y = batchSize;
    //should be ≈ smCount * 6 / batchSize
    numBlocks.x = min((polynomCount + blockSize - 1) * outputDim / blockSize, 160);

    const int maxDepth = 12;
    const int K = 16;

    ExpProbPolynomBackwardImpl<maxDepth, blockSize, K> <<<numBlocks, blockSize>>>(lambda, features, featuresCount, outDer, outputDim, featureIds, conditions, values, polynomOffsets, polynomCount, featuresDer);

}

// }}}




// SigmoidProb Polynom {{{

__global__ void SigmoidProbPolynomProbsImpl(
        const float* features,
        int batchSize,
        const int* splits,
        const float* conditions,
        const int* polynomOffsets,
        int polynomCount,
        float lambda,
        float* probs) {
    if (threadIdx.x < batchSize) {
        int polynomId = blockIdx.x;

        features +=  threadIdx.x;
        probs += threadIdx.x;

        while (polynomId < polynomCount) {
            int offset = polynomOffsets[polynomId];
            int nextOffset = polynomOffsets[polynomId + 1];
            const int depth = nextOffset - offset;

//            bool isTrue = true;
            float logProb = 0;
            for (int i = 0; i < depth; ++i) {
                const int f = __ldg(splits + offset + i);
                const float c = __ldg(conditions + offset + i);
                const float x = __ldg(features + f * batchSize);
                const float val = -lambda * (x - c);
//                isTrue = x <= c? false : isTrue;
                const float expVal = 1.0f + expf(val);

//            p( split = 1) = 1.0 / (1.0 + exp(-(x - c)))
//            c = 0, x= inf, p = 1.0 / (1.0 + exp(-inf) = 0
//            log(p) = -log(1.0 + exp(-(x - c))
                const float isTrueLogProb = isfinite(expVal) ? log(expVal) : val;
                logProb -= isTrueLogProb;
            }
            const float prob = expf(logProb);
//            const float prob = isTrue ? 1 : 0;//exp(logProb);
            probs[polynomId * batchSize] = prob;
            polynomId += gridDim.x;
        }
    }
}

//batch size should be equal to BlockSize
//we need to reduce polynoms for each output dim
__global__ void SigmoidProbPolynomForwardImpl(
        const float* probs,
        int batchSize,
        const float* values,
        int polynomCount,
        int outputDim,
        float* out) {

    //out: batch_elem0 dim0, dim1, dimk batch_elem1 dim0 dim1 dimk
    //so threads
    int polynomId = blockIdx.x;
    const int dimId = blockIdx.y;

    int tid = threadIdx.x;
    if (tid >= batchSize) {
        return;
    }

    float sum = 0;
    probs += threadIdx.x;
    values += dimId;

    while (polynomId < polynomCount) {
        const float polynomProb = __ldg(probs + polynomId * batchSize);
        const float out = __ldg(values + polynomId * outputDim);
        sum += polynomProb * out;
        polynomId += gridDim.x;
    }

    atomicAdd(out + dimId * batchSize + threadIdx.x, sum);
}

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
) {
    const int blockSize = batchSize;
    const int numBlocks = min(polynomCount, 1000);
    assert(batchSize < 2048);
    assert(numBlocks);

    SigmoidProbPolynomProbsImpl << < numBlocks, blockSize >>> (features, batchSize, splits, conditions, polynomOffsets, polynomCount, lambda, tempProbs);

    dim3 forwardBlocks;
    forwardBlocks.z = 1;
    forwardBlocks.y = outDim;
    forwardBlocks.x = min(polynomCount, 512);
    SigmoidProbPolynomForwardImpl << < forwardBlocks, batchSize >> > (tempProbs, batchSize, values, polynomCount, outDim, output);
}

/*
 * Here layout is not the same as in forward pass
 * BlockSize = 256, MaxDepth = 8, K = 24
 * should give 50% occupancy, this should be enough
 */
template <int MaxDepth, int BlockSize, int K>
__global__ void SigmoidProbPolynomBackwardImpl(float lambda,
                                           const float* features,
                                           int featuresCount,
                                           const float* outDer,
                                           int outputDim,
                                           const int* featureIds,
                                           const float* conditions,
                                           const float* values,
                                           const int* polynomOffsets,
                                           int polynomCount,
                                           float* featuresDer) {
    const int sampleId = blockIdx.y;

    features += sampleId * featuresCount;
    featuresDer += sampleId * featuresCount;
    outDer += sampleId * outputDim;


    //out: batch_elem0 dim0, dim1, dimk batch_elem1 dim0 dim1 dimk
    //so threads

    __shared__ float localFeaturesDer[BlockSize * K];

    for (int i = threadIdx.x; i < BlockSize * K; i += BlockSize) {
        localFeaturesDer[i] = 0;
    }
    __syncthreads();


    const int alignedFeaturesCount = ((featuresCount + BlockSize - 1) / BlockSize) * BlockSize;

    const int memoryBlocks = (BlockSize * K) / alignedFeaturesCount;
    const int memoryBlockId = threadIdx.x % memoryBlocks;


    int polynomId = blockIdx.x * blockDim.x + threadIdx.x;

    while (polynomId < polynomCount) {
        const int offset = polynomOffsets[polynomId];
        const int nextOffset = polynomOffsets[polynomId + 1];
        const int depth = nextOffset - offset;

        if (depth != 0) {

            float logProbs[MaxDepth];
            short fids[MaxDepth];
            float totalLogProb = 0;

#pragma unroll
            for (int i = 0; i < MaxDepth; ++i) {
                if (i < depth) {
                    const int f = __ldg(featureIds + i + offset);
                    fids[i] = f;
                    const float c = __ldg(conditions + i + offset);
                    const float x = __ldg(features + f);
                    const float val = -lambda * (x - c);
                    const float expVal = 1.0f + exp(val);
                    logProbs[i] = -(isfinite(expVal) ? log(expVal) : val);
                    totalLogProb += logProbs[i];
                }
            }

            const float p = exp(totalLogProb);

            //featureDerivative is outputDer * total value before monom * monom derivative
            float derMultiplier = 0;
#pragma unroll 10
            for (int dim = 0; dim < outputDim; ++dim) {
                derMultiplier += __ldg(values + polynomId * outputDim + dim) * __ldg(outDer + dim);
            }

#pragma unroll
            for (int i = 0; i < MaxDepth; ++i) {
                if (i < depth) {
                    const int f = fids[i];
                    const float featureProb = exp(logProbs[i]);
                    const float monomDer = p * (1.0 - featureProb);
                    const float featureDer = monomDer * derMultiplier;
                    //atomics in shared memory, pretty fast on pascal+ hardware
                    atomicAdd(localFeaturesDer + memoryBlocks * f + memoryBlockId, featureDer);
                }
            }
        }
        polynomId += gridDim.x * blockDim.x;
    }


    __syncthreads();

    //outputDim = 1024 => memoryBlocks = 6
    for (int f = threadIdx.x; f < featuresCount; f += BlockSize) {
        float der = 0;

#pragma unroll
        for (int k = 0; k < memoryBlocks; ++k) {
            der += localFeaturesDer[f * memoryBlocks + k];
        }
        atomicAdd(featuresDer + f,  der);
    }
}

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
                            float* featuresDer) {

    const int blockSize = 256;
    dim3 numBlocks;
    numBlocks.z = 1;
    numBlocks.y = batchSize;
    //should be ≈ smCount * 6 / batchSize
    numBlocks.x = min((polynomCount + blockSize - 1) * outputDim / blockSize, 160);

    const int maxDepth = 12;
    const int K = 16;

    SigmoidProbPolynomBackwardImpl<maxDepth, blockSize, K> <<<numBlocks, blockSize>>>(lambda, features, featuresCount, outDer, outputDim, featureIds, conditions, values, polynomOffsets, polynomCount, featuresDer);

}

// }}}



// TODO move linear polynom to a separate file

// LinearPolynom {{{

__global__ void LinearPolynomProbsImpl(
        const float* features,
        int batchSize,
        const int* splits,
        const float* conditions,
        const int* polynomOffsets,
        int polynomCount,
        float lambda,
        float* probs,
        const int* origFIds) {
    if (threadIdx.x < batchSize) {
        int polynomId = blockIdx.x;

        features += threadIdx.x;
        probs += threadIdx.x;

        while (polynomId < polynomCount) {
            int offset = polynomOffsets[polynomId];
            int nextOffset = polynomOffsets[polynomId + 1];
            const int depth = nextOffset - offset;
            const int origFId = origFIds[polynomId];

            bool zeroProb = false;
            for (int i = 0; i < depth; ++i) {
                if (zeroProb) {
                    continue;
                }

                const float c = __ldg(conditions + offset + i);

                const int f = __ldg(splits + offset + i);
                const float x = __ldg(features + f * batchSize);

                if (x <= c) {
                    zeroProb = true;
                }
            }

            float prob = 0.0f;
            if (!zeroProb) {
                prob = __ldg(features + origFId * batchSize);
            }

            probs[polynomId * batchSize] = prob;
            polynomId += gridDim.x;
        }
    }
}

//batch size should be equal to BlockSize
//we need to reduce polynoms for each output dim
__global__ void LinearPolynomForwardImpl(
        const float* probs,
        int batchSize,
        const float* values,
        int polynomCount,
        int outputDim,
        float* out) {

    //out: batch_elem0 dim0, dim1, dimk batch_elem1 dim0 dim1 dimk
    //so threads
    int polynomId = blockIdx.x;
    const int dimId = blockIdx.y;

    int tid = threadIdx.x;
    if (tid >= batchSize) {
        return;
    }

    float sum = 0;
    probs += threadIdx.x;
    values += dimId;

    while (polynomId < polynomCount) {
        const float polynomProb = __ldg(probs + polynomId * batchSize); // includes x
        const float v = __ldg(values + polynomId * outputDim);
        sum += polynomProb * v;
        polynomId += gridDim.x;
    }

    atomicAdd(out + dimId * batchSize + threadIdx.x, sum);
}

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
        const int* origFIds
) {
    const int blockSize = batchSize;
    const int numBlocks = min(polynomCount, 1000);
    assert(batchSize < 2048);
    assert(numBlocks);

    LinearPolynomProbsImpl << < numBlocks, blockSize >>> (features, batchSize, splits, conditions, polynomOffsets, polynomCount, lambda, tempProbs, origFIds);

    dim3 forwardBlocks;
    forwardBlocks.z = 1;
    forwardBlocks.y = outDim;
    forwardBlocks.x = min(polynomCount, 512);
    LinearPolynomForwardImpl << < forwardBlocks, batchSize >> > (tempProbs, batchSize, values, polynomCount, outDim, output);
}

/*
 * Here layout is not the same as in forward pass
 * BlockSize = 256, MaxDepth = 8, K = 24
 * should give 50% occupancy, this should be enough
 */
template <int MaxDepth, int BlockSize, int K>
__global__ void LinearPolynomBackwardImpl(float lambda,
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
                                          const int* origFIds) {
    const int sampleId = blockIdx.y;

    features += sampleId * featuresCount;
    featuresDer += sampleId * featuresCount;
    outDer += sampleId * outputDim;


    //out: batch_elem0 dim0, dim1, dimk batch_elem1 dim0 dim1 dimk
    //so threads

    __shared__ float localFeaturesDer[BlockSize * K];

    for (int i = threadIdx.x; i < BlockSize * K; i += BlockSize) {
        localFeaturesDer[i] = 0;
    }
    __syncthreads();


    const int alignedFeaturesCount = ((featuresCount + BlockSize - 1) / BlockSize) * BlockSize;

    const int memoryBlocks = (BlockSize * K) / alignedFeaturesCount;
    const int memoryBlockId = threadIdx.x % memoryBlocks;


    int polynomId = blockIdx.x * blockDim.x + threadIdx.x;

    while (polynomId < polynomCount) {
        const int offset = polynomOffsets[polynomId];
        const int nextOffset = polynomOffsets[polynomId + 1];
        const int depth = nextOffset - offset;
        const int origFId = origFIds[polynomId];

        // TODO for now we treat origFId=0 as bias
        if (origFId != 0) {

            bool isZero = false;

#pragma unroll
            for (int i = 0; i < MaxDepth; ++i) {
                if (isZero) {
                    continue;
                }
                if (i < depth) {
                    const float c = __ldg(conditions + i + offset);
                    const int f = __ldg(featureIds + i + offset);
                    const float x = __ldg(features + f);
                    if (x <= c) {
                        isZero = true;
                    }
                }
            }

            //featureDerivative is outputDer * total value before monom * monom derivative
            float derMultiplier = 0;
            if (!isZero) {
#pragma unroll 10
                for (int dim = 0; dim < outputDim; ++dim) {
                    derMultiplier += __ldg(values + polynomId * outputDim + dim) * __ldg(outDer + dim);
                }
            }

            const float featureDer = derMultiplier * __ldg(features + origFId);
            atomicAdd(localFeaturesDer + memoryBlocks * origFId + memoryBlockId, derMultiplier);
        }
        polynomId += gridDim.x * blockDim.x;
    }


    __syncthreads();

    //outputDim = 1024 => memoryBlocks = 6
    for (int f = threadIdx.x; f < featuresCount; f += BlockSize) {
        float der = 0;

#pragma unroll
        for (int k = 0; k < memoryBlocks; ++k) {
            der += localFeaturesDer[f * memoryBlocks + k];
        }
        atomicAdd(featuresDer + f,  der);
    }
}

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
                           const int* origFIds) {

    const int blockSize = 256;
    dim3 numBlocks;
    numBlocks.z = 1;
    numBlocks.y = batchSize;
    //should be ≈ smCount * 6 / batchSize
    numBlocks.x = min((polynomCount + blockSize - 1) * outputDim / blockSize, 160);

    const int maxDepth = 12;
    const int K = 16;

    LinearPolynomBackwardImpl<maxDepth, blockSize, K> <<<numBlocks, blockSize>>>(lambda, features, featuresCount, outDer, outputDim, featureIds, conditions, values, polynomOffsets, polynomCount, featuresDer, origFIds);
}


// }}}
