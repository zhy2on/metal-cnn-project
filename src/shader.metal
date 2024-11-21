// shader.metal
#include <metal_stdlib>
using namespace metal;

constant int BATCH_SIZE = 64;

kernel void convolution_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* network [[buffer(2)]],
    device const int* params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const int inDim = params[0];
    const int outDim = params[1];
    const int nbyn = params[2];
    const int imageOffset = params[3];
    const int networkOffset = params[4];
    
    const int col = tid.x;
    const int row = tid.y;
    const int batch_outNeuron = tid.z;
    const int batch = batch_outNeuron / outDim;
    const int outNeuron = batch_outNeuron % outDim;
    
    if (row >= nbyn || col >= nbyn || batch >= BATCH_SIZE || outNeuron >= outDim) {
        return;
    }
    
    const int batchOffset = batch * inDim * nbyn * nbyn;
    float sum = 0.0f;
    
    const device float* filters = network + networkOffset;
    const device float* biases = filters + 3 * 3 * inDim * outDim;
    
    for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        const device float* currentInput = input + imageOffset + batchOffset + inNeuron * nbyn * nbyn;
        const device float* filter = filters + (outNeuron * inDim + inNeuron) * 9;
        
        for (int i = 0; i < 3; ++i) {
            int y = row + i - 1;
            if (y >= 0 && y < nbyn) {
                for (int j = 0; j < 3; ++j) {
                    int x = col + j - 1;
                    if (x >= 0 && x < nbyn) {
                        sum += currentInput[y * nbyn + x] * filter[i * 3 + j];
                    }
                }
            }
        }
    }
    
    sum += biases[outNeuron];
    sum = max(0.0f, sum);  // ReLU activation
    
    const int outputOffset = batch * outDim * nbyn * nbyn;
    output[outputOffset + outNeuron * nbyn * nbyn + row * nbyn + col] = sum;
}

kernel void max_pooling_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const int* params [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const int DIM = params[0];
    const int outNbyn = params[1];
    
    const int n = tid.x;
    const int idx = tid.y;
    const int batch = tid.z;
    
    if (batch >= BATCH_SIZE || n >= DIM || idx >= outNbyn * outNbyn) {
        return;
    }
    
    const int inNbyn = outNbyn * 2;
    
    const int row = idx / outNbyn;
    const int col = idx % outNbyn;
    
    const int inRow = row * 2;
    const int inCol = col * 2;
    
    const int input_offset = batch * DIM * inNbyn * inNbyn;
    const int output_offset = batch * DIM * outNbyn * outNbyn;
    
    float maxVal = input[input_offset + n * inNbyn * inNbyn + inRow * inNbyn + inCol];
    maxVal = max(maxVal, input[input_offset + n * inNbyn * inNbyn + inRow * inNbyn + inCol + 1]);
    maxVal = max(maxVal, input[input_offset + n * inNbyn * inNbyn + (inRow + 1) * inNbyn + inCol]);
    maxVal = max(maxVal, input[input_offset + n * inNbyn * inNbyn + (inRow + 1) * inNbyn + inCol + 1]);
    
    output[output_offset + n * outNbyn * outNbyn + row * outNbyn + col] = maxVal;
}

kernel void fc_layer_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* network [[buffer(2)]],
    device const int* params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const int inDim = params[0];
    const int outDim = params[1];
    const int networkOffset = params[2];
    
    const int outIdx = tid.x;
    const int batch = tid.y;
    
    if (outIdx >= outDim || batch >= BATCH_SIZE) {
        return;
    }
    
    const device float* weights = network + networkOffset;
    const device float* biases = weights + inDim * outDim;
    
    float sum = 0.0f;
    for (int inIdx = 0; inIdx < inDim; ++inIdx) {
        sum += input[batch * inDim + inIdx] * weights[outIdx * inDim + inIdx];
    }
    
    sum += biases[outIdx];
    output[batch * outDim + outIdx] = max(0.0f, sum);  // ReLU activation
}
