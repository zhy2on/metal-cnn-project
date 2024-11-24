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
    
    // OpenCL 버전과 동일한 인덱싱 방식 적용
    const int idx = tid.x;
    const int outNeuron = tid.y;
    const int batch = tid.z;
    
    const int row = idx / nbyn;
    const int col = idx % nbyn;
    
    if (row >= nbyn || col >= nbyn || outNeuron >= outDim || batch >= BATCH_SIZE) {
        return;
    }
    
    float sum = 0.0f;
    const device float* filters = network + networkOffset;
    const device float* biases = filters + 3 * 3 * inDim * outDim;
    
    for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        const device float* currentInput = input + imageOffset + batch * inDim * nbyn * nbyn + inNeuron * nbyn * nbyn;
        const device float* filter = filters + (outNeuron * inDim + inNeuron) * 9;
        
        int x0 = col - 1, x1 = col, x2 = col + 1;
        int y0 = row - 1, y1 = row, y2 = row + 1;
        
        if (y0 >= 0 && y0 < nbyn) {
            if (x0 >= 0 && x0 < nbyn) sum += currentInput[y0 * nbyn + x0] * filter[0];
            if (x1 >= 0 && x1 < nbyn) sum += currentInput[y0 * nbyn + x1] * filter[1];
            if (x2 >= 0 && x2 < nbyn) sum += currentInput[y0 * nbyn + x2] * filter[2];
        }
        
        if (y1 >= 0 && y1 < nbyn) {
            if (x0 >= 0 && x0 < nbyn) sum += currentInput[y1 * nbyn + x0] * filter[3];
            if (x1 >= 0 && x1 < nbyn) sum += currentInput[y1 * nbyn + x1] * filter[4];
            if (x2 >= 0 && x2 < nbyn) sum += currentInput[y1 * nbyn + x2] * filter[5];
        }
        
        if (y2 >= 0 && y2 < nbyn) {
            if (x0 >= 0 && x0 < nbyn) sum += currentInput[y2 * nbyn + x0] * filter[6];
            if (x1 >= 0 && x1 < nbyn) sum += currentInput[y2 * nbyn + x1] * filter[7];
            if (x2 >= 0 && x2 < nbyn) sum += currentInput[y2 * nbyn + x2] * filter[8];
        }
    }
    
    sum += biases[outNeuron];
    sum = max(0.0f, sum);  // ReLU activation
    
    const int batch_offset = batch * outDim * nbyn * nbyn;
    output[batch_offset + outNeuron * nbyn * nbyn + row * nbyn + col] = sum;
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
