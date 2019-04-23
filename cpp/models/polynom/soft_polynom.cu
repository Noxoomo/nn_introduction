#include <cuda_runtime.h>
#if __CUDA_ARCH__ < 350
template <typename T>
__forceinline__ __device__ T __ldg(const T* data) {
    return data[0];
}
#endif

__forceinline__ __device__  float Sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


__forceinline__ __device__  float SigmoidDer(float x) {
    const float p = 1.0f / (1.0f + expf(-x));
    return p * (1.0f - p);
}


__global__ void PolynomProbsImpl(const float* features,
                                 int fCount,
                                 int batchSize,
                                 const int* splits,
                                 const float* conditions,
                                 const int* polynomOffsets,
                                 const int* polynomDepth,
                                 int polynomCount,
                                 float* probs) {
    if (threadIdx.x < batchSize) {
        int polynomId = blockIdx.x;

        while (polynomId < polynomCount) {
            const int depth = polynomDepth[polynomId];
            int offset = polynomOffsets[polynomId];

            conditions += offset;
            splits += offset;

            float logProb = 0;
            for (int i = 0; i < depth; ++i) {
                const int f = __ldg(splits + i);
                const float c = __ldg(conditions + i);
                const float x = __ldg(features + f * batchSize + threadIdx.x);
                const float val = -(x - c);
                const float expVal = 1.0f + exp(val);

//            p( split = 1) = 1.0 / (1.0 + exp(-(x - c)))
//            c = 0, x= inf, p = 1.0 / (1.0 + exp(-inf) = 0
//            log(p) = -log(1.0 + exp(-(x - c))
                const float isTrueLogProb = isfinite(expVal) ? log(expVal) : val;
                logProb -= isTrueLogProb;
            }
            const float prob = exp(logProb);
            probs[polynomId * batchSize + threadIdx.x] = prob;
            polynomId += gridDim.x;
        }
    }
}


//batch size should be equal to BlockSize
//we need to reduce polynoms for each output dim
__global__ void PolynomForwardImpl(const float* probs,
                                   int batchSize,
                                   const float* values,
                                   int polynomCount,
                                   int outputDim,
                                   float* out) {
    //out: batch_elem0 dim0, dim1, dimk batch_elem1 dim0 dim1 dimk
    //so threads


    const size_t totalElems = outputDim * batchSize * polynomCount;
    const int flatElems = outputDim * batchSize;

    const int stripeSize = gridDim.x * flatElems;

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    const int elemId = tid % flatElems;
    const int dimId = elemId % outputDim;
    const int batchId = elemId / outputDim;

    /*
     * batch size should be 2^k (128/256/512), so this will be always false
     */
    if (tid >= stripeSize) {
        return;
    }

    float sum = 0;

    while (tid < totalElems) {
        const int polynomId = tid / flatElems;
        const float polynomProb = __ldg(probs + polynomId);
        const float out = __ldg(values + polynomId * outputDim + dimId);
        sum += polynomProb * out;

        tid += stripeSize;
    }
    atomicAdd(out + batchId * outputDim + dimId, sum);
}


//


void PolynomForward(const float* features,
    int fCount,
    int batchSize,
    const int* splits,
    const float* conditions,
    const int* polynomOffsets,
    const int* polynomDepth,
    int polynomCount,
    int outputDim,
    float* out
    ) {

}




/*
 * Here layout is not the same as in forward pass
 * BlockSize = 256, MaxDepth = 6, K = 24
 * should give 50% occupancy, this should be enough
 */
template <int MaxDepth, int BlockSize, int K>
__global__ void PolynomBackwardImpl(const float* features,
                                    int featuresCount,
                                    const float* outDer,
                                    int outputDim,
                                    const float* leafSum,
                                    int* polynomDepths,
                                    int* polynomOffset,
                                    int* featureIds,
                                    float* conditions,
                                    int polynomCount,
                                    float* out) {
    const int sampleId = blockIdx.y;
    features += sampleId * featuresCount;
    out += sampleId * featuresCount;

    outDer += sampleId * outputDim;
//    float outputDer = 0;
//    for (int dim = 0; dim < outputDim; ++dim) {
//        outputDer += outDer[dim];
//    }

    //out: batch_elem0 dim0, dim1, dimk batch_elem1 dim0 dim1 dimk
    //so threads

    __shared__ float localFeaturesDer[BlockSize * K];
    for (int i = threadIdx.x; i < BlockSize * K; i += BlockSize) {
        localFeaturesDer[i] = 0;
    }
    __syncthreads();
    const int alignedFeaturesCount = ((featuresCount + BlockSize - 1) / BlockSize) * BlockSize;
    const int memoryBlocks = BlockSize * K / alignedFeaturesCount;
    const int memoryBlockId = threadIdx.x % memoryBlocks;


    int polynomId = blockIdx.x * gridDim.x + threadIdx.x;

    while (polynomId < polynomCount) {
        const int depth = polynomDepths[polynomId];
        int offset = polynomOffset[polynomId];


        float logOneMinusProb[MaxDepth];
        short fids[MaxDepth];
        float totalLogProb = 0;

        #pragma unroll
        for (int i = 0; i < MaxDepth; ++i) {
            if (i < depth) {
                const int f = __ldg(featureIds + i + offset);
                fids[i] = f;
                const float c = __ldg(conditions + i + offset);
                const float x = __ldg(features + f);
                const float val = -(x - c);
                const float expVal = 1.0f + exp(val);
                const float isTrueLogProb = (isfinite(expVal) ? log(expVal) : val);
                totalLogProb += isTrueLogProb;
                logOneMinusProb[i] = val - isTrueLogProb;
            }
        }

        //featureDerivative is outputDer * total value before monom * monom derivative
        float derMultiplier  = 0;
        for (int dim = 0; dim < outputDim; ++dim) {
            derMultiplier += __ldg(leafSum + polynomId * outputDim + dim) * __ldg(outDer + dim);
        }

        #pragma unroll
        for (int i = 0; i < MaxDepth; ++i) {
            if (i < depth) {
                const int f = fids[i];
                const int featureDer = exp(totalLogProb + logOneMinusProb[i]) * derMultiplier;
                //atomics in shared memory, pretty fast on pascal+ hardware
                atomicAdd(localFeaturesDer + memoryBlocks * f + memoryBlockId, featureDer);
            }
        }
        polynomId += gridDim.x * blockDim.x;
    }


    __syncthreads();

    //outputDim = 1024 => memoryBlocks = 6
    for (int i = threadIdx.x; i < featuresCount; i += BlockSize) {
        float der = 0;

        for (int k = 0; k < memoryBlocks; ++k) {
            der += localFeaturesDer[i * memoryBlocks + k];
        }
        atomicAdd(out + i,  localFeaturesDer[i * memoryBlocks + i]);
    }
}

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
                     cudaStream_t stream) {

    const int blockSize = 256;
    dim3 numBlocks;
    numBlocks.z = 1;
    numBlocks.y = batchSize;
    //should be ≈ smCount * 6 / batchSize
    numBlocks.x = (polynomCount + blockSize - 1) / blockSize;

    const int maxDepth = 6;
    const int K = 16;
    PolynomBackwardImpl<maxDepth, blockSize, K> <<<numBlocks, blockSize, 0, stream >>>(features, featuresCount, outDer, outputDim,
        leafSum, polynomDepths, polynomOffset, featureIds, conditions, polynomCount, out);

}
