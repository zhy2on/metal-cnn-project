#include <metal_stdlib>
using namespace metal;

kernel void convolution_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* filter [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device const int* params [[buffer(4)]],
    uint3 id [[thread_position_in_grid]]
) {
    const int outNeuron = id.z;
    const int row = id.y;
    const int col = id.x;
    
    const int inDim = params[0];
    const int outDim = params[1];
    const int nbyn = params[2];
    const int offset = nbyn * nbyn;
    
    float sum = 0.0f;
    
    for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        for (int fRow = 0; fRow < 3; ++fRow) {
            for (int fCol = 0; fCol < 3; ++fCol) {
                int x = col + fCol - 1;
                int y = row + fRow - 1;
                
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                    const int inputIdx = inNeuron * offset + y * nbyn + x;
                    const int filterIdx = (outNeuron * inDim + inNeuron) * 9 + fRow * 3 + fCol;
                    sum += input[inputIdx] * filter[filterIdx];
                }
            }
        }
    }
    
    const int outIdx = outNeuron * offset + row * nbyn + col;
    sum += bias[outNeuron];
    output[outIdx] = sum > 0.0f ? sum : 0.0f; // ReLU
}
