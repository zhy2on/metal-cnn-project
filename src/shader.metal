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
    
    const int idx = tid.x;
    const int outNeuron = tid.y;
    const int batch = tid.z;
    
    if (outNeuron >= outDim || batch >= BATCH_SIZE) return;
    
    const int row = idx / nbyn;
    const int col = idx % nbyn;
    
    if (row >= nbyn || col >= nbyn) return;
    
    const device float* filters = network + networkOffset;
    const device float* biases = filters + 3 * 3 * inDim * outDim;
    float sum = 0.0f;
    
    const int batch_offset = batch * inDim * nbyn * nbyn;
    const int filter_offset = outNeuron * inDim * 9;
    
    const int center_idx = row * nbyn + col;
    const int top_idx = ((row > 0) ? (row-1) : row) * nbyn + col;
    const int bottom_idx = ((row < nbyn-1) ? (row+1) : row) * nbyn + col;
    
    for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        const device float* currentInput = input + imageOffset + batch_offset + inNeuron * nbyn * nbyn;
        const device float* filter = filters + filter_offset + inNeuron * 9;
        
        // Center row
        const float center = currentInput[center_idx];
        const float left = (col > 0) ? currentInput[center_idx - 1] : 0;
        const float right = (col < nbyn-1) ? currentInput[center_idx + 1] : 0;
        
        // Top row
        const float top = (row > 0) ? currentInput[top_idx] : 0;
        const float topLeft = (row > 0 && col > 0) ? currentInput[top_idx - 1] : 0;
        const float topRight = (row > 0 && col < nbyn-1) ? currentInput[top_idx + 1] : 0;
        
        // Bottom row
        const float bottom = (row < nbyn-1) ? currentInput[bottom_idx] : 0;
        const float bottomLeft = (row < nbyn-1 && col > 0) ? currentInput[bottom_idx - 1] : 0;
        const float bottomRight = (row < nbyn-1 && col < nbyn-1) ? currentInput[bottom_idx + 1] : 0;
        
        sum += topLeft * filter[0] + top * filter[1] + topRight * filter[2]
             + left * filter[3] + center * filter[4] + right * filter[5]
             + bottomLeft * filter[6] + bottom * filter[7] + bottomRight * filter[8];
    }
    
    sum += biases[outNeuron];
    sum = max(0.0f, sum);
    
    const int output_offset = batch * outDim * nbyn * nbyn;
    output[output_offset + outNeuron * nbyn * nbyn + center_idx] = sum;
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
