#include <metal_stdlib>
using namespace metal;

kernel void convolution(
    const device float* inputs [[buffer(0)]],
    device float* outputs [[buffer(1)]],
    const device float* networks [[buffer(2)]],
    constant int& inDim [[buffer(3)]],
    constant int& outDim [[buffer(4)]],
    constant int& nbyn [[buffer(5)]],
    constant int& image_offset [[buffer(6)]],
    constant int& filter_offset [[buffer(7)]],
    uint3 position [[thread_position_in_grid]]
) {
    const int idx = position.x;
    const int outNeuron = position.y;
    const int batch = position.z;
    
    // Early return for out of bounds
    if (outNeuron >= outDim || batch >= 500) return;  // BATCH_SIZE = 500
    
    const int row = idx / nbyn;
    const int col = idx % nbyn;
    
    if (row >= nbyn || col >= nbyn) return;
    
    // Pre-calculate offsets
    const device float* filters = networks + filter_offset;
    const device float* biases = filters + 3 * 3 * inDim * outDim;
    
    float sum = 0.0f;
    const int batch_offset = batch * inDim * nbyn * nbyn;
    const int center_idx = row * nbyn + col;
    
    #pragma unroll(3)
    for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        const device float* current_input = inputs + image_offset + batch_offset + inNeuron * nbyn * nbyn;
        const device float* filter = filters + (outNeuron * inDim + inNeuron) * 9;
        
        // Center pixels
        float center = current_input[center_idx];
        float left = (col > 0) ? current_input[center_idx - 1] : 0.0f;
        float right = (col < nbyn-1) ? current_input[center_idx + 1] : 0.0f;
        
        // Top pixels
        int top_idx = (row > 0) ? center_idx - nbyn : center_idx;
        float top = (row > 0) ? current_input[top_idx] : 0.0f;
        float top_left = (row > 0 && col > 0) ? current_input[top_idx - 1] : 0.0f;
        float top_right = (row > 0 && col < nbyn-1) ? current_input[top_idx + 1] : 0.0f;
        
        // Bottom pixels
        int bottom_idx = (row < nbyn-1) ? center_idx + nbyn : center_idx;
        float bottom = (row < nbyn-1) ? current_input[bottom_idx] : 0.0f;
        float bottom_left = (row < nbyn-1 && col > 0) ? current_input[bottom_idx - 1] : 0.0f;
        float bottom_right = (row < nbyn-1 && col < nbyn-1) ? current_input[bottom_idx + 1] : 0.0f;
        
        sum += top_left * filter[0] + top * filter[1] + top_right * filter[2]
             + left * filter[3] + center * filter[4] + right * filter[5]
             + bottom_left * filter[6] + bottom * filter[7] + bottom_right * filter[8];
    }
    
    sum += biases[outNeuron];
    sum = fmax(0.0f, sum);  // ReLU activation
    
    int out_idx = batch * outDim * nbyn * nbyn + outNeuron * nbyn * nbyn + center_idx;
    outputs[out_idx] = sum;
}

kernel void max_pooling(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant int& DIM [[buffer(2)]],
    constant int& nbyn [[buffer(3)]],
    uint3 position [[thread_position_in_grid]]
) {
    const int n = position.x;
    const int idx = position.y;
    const int batch = position.z;
    
    // Calculate output dimensions
    const int out_dim = nbyn / 2;
    
    if (batch >= 500 || n >= DIM || idx >= out_dim * out_dim) return;
    
    // Calculate input/output positions
    const int row = idx / out_dim;
    const int col = idx % out_dim;
    
    const int in_row = row * 2;
    const int in_col = col * 2;
    
    // Calculate memory offsets
    const int in_offset = batch * DIM * nbyn * nbyn + n * nbyn * nbyn;
    const int out_offset = batch * DIM * out_dim * out_dim + n * out_dim * out_dim;
    
    // Get 2x2 window values
    float val1 = input[in_offset + in_row * nbyn + in_col];
    float val2 = input[in_offset + in_row * nbyn + (in_col + 1)];
    float val3 = input[in_offset + (in_row + 1) * nbyn + in_col];
    float val4 = input[in_offset + (in_row + 1) * nbyn + (in_col + 1)];
    
    // Find max value
    float max_val = fmax(fmax(val1, val2), fmax(val3, val4));
    
    // Write result
    output[out_offset + row * out_dim + col] = max_val;
}

kernel void fc_layer(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    const device float* network [[buffer(2)]],
    constant int& inDim [[buffer(3)]],
    constant int& outDim [[buffer(4)]],
    constant int& offset [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]
) {
    const int outIdx = position.x;
    const int batch = position.y;
    if (outIdx >= outDim || batch >= 500) return;
    
    // Get weights and bias pointers
    const device float* weights = network + offset;
    const device float* bias = weights + inDim * outDim;
    
    // Calculate dot product
    float sum = 0.0f;
    int input_offset = batch * inDim;
    int weight_offset = outIdx * inDim;
    
    #pragma unroll(8)
    for (int i = 0; i < inDim; ++i) {
        sum += input[input_offset + i] * weights[weight_offset + i];
    }
    
    // Add bias and apply ReLU
    sum += bias[outIdx];
    output[batch * outDim + outIdx] = fmax(0.0f, sum);
}
