#include <metal_stdlib>
using namespace metal;

kernel void convolution(
    const device float* inputs [[buffer(0)]],
    device float* outputs [[buffer(1)]],
    const device float* networks [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    constant int& nbyn [[buffer(5)]],
    constant int& image_network_offset [[buffer(6)]],
    constant int& network_network_offset [[buffer(7)]],
    uint3 position [[thread_position_in_grid]]
) {
    const int idx = position.x;
    const int outNeuron = position.y;
    const int batch = position.z;
    
    // Early return for out of bounds
    if (outNeuron >= out_dim || batch >= 500) return;  // BATCH_SIZE = 500
    
    const int row = idx / nbyn;
    const int col = idx % nbyn;
    
    if (row >= nbyn || col >= nbyn) return;
    
    // Pre-calculate network_offsets
    const device float* filters = networks + network_network_offset;
    const device float* biases = filters + 3 * 3 * in_dim * out_dim;
    
    float sum = 0.0f;
    const int batch_network_offset = batch * in_dim * nbyn * nbyn;
    const int center_idx = row * nbyn + col;
    
    #pragma unroll(3)
    for (int inNeuron = 0; inNeuron < in_dim; ++inNeuron) {
        const device float* current_inputs = inputs + image_network_offset + batch_network_offset + inNeuron * nbyn * nbyn;
        const device float* filter = filters + (outNeuron * in_dim + inNeuron) * 9;
        
        // Center pixels
        float center = current_inputs[center_idx];
        float left = (col > 0) ? current_inputs[center_idx - 1] : 0.0f;
        float right = (col < nbyn-1) ? current_inputs[center_idx + 1] : 0.0f;
        
        // Top pixels
        int top_idx = (row > 0) ? center_idx - nbyn : center_idx;
        float top = (row > 0) ? current_inputs[top_idx] : 0.0f;
        float top_left = (row > 0 && col > 0) ? current_inputs[top_idx - 1] : 0.0f;
        float top_right = (row > 0 && col < nbyn-1) ? current_inputs[top_idx + 1] : 0.0f;
        
        // Bottom pixels
        int bottom_idx = (row < nbyn-1) ? center_idx + nbyn : center_idx;
        float bottom = (row < nbyn-1) ? current_inputs[bottom_idx] : 0.0f;
        float bottom_left = (row < nbyn-1 && col > 0) ? current_inputs[bottom_idx - 1] : 0.0f;
        float bottom_right = (row < nbyn-1 && col < nbyn-1) ? current_inputs[bottom_idx + 1] : 0.0f;
        
        sum += top_left * filter[0] + top * filter[1] + top_right * filter[2]
             + left * filter[3] + center * filter[4] + right * filter[5]
             + bottom_left * filter[6] + bottom * filter[7] + bottom_right * filter[8];
    }
    
    sum += biases[outNeuron];
    sum = fmax(0.0f, sum);  // ReLU activation
    
    int out_idx = batch * out_dim * nbyn * nbyn + outNeuron * nbyn * nbyn + center_idx;
    outputs[out_idx] = sum;
}

kernel void max_pooling(
    const device float* inputs [[buffer(0)]],
    device float* outputs [[buffer(1)]],
    constant int& in_dim [[buffer(2)]],
    constant int& nbyn [[buffer(3)]],
    uint3 position [[thread_position_in_grid]]
) {
    const int n = position.x;
    const int idx = position.y;
    const int batch = position.z;
    
    // Calculate outputs dimensions
    const int out_dim = nbyn / 2;
    
    if (batch >= 500 || n >= in_dim || idx >= out_dim * out_dim) return;
    
    // Calculate inputs/outputs positions
    const int row = idx / out_dim;
    const int col = idx % out_dim;
    
    const int in_row = row * 2;
    const int in_col = col * 2;
    
    // Calculate memory network_offsets
    const int in_network_offset = batch * in_dim * nbyn * nbyn + n * nbyn * nbyn;
    const int out_network_offset = batch * in_dim * out_dim * out_dim + n * out_dim * out_dim;
    
    // Get 2x2 window values
    float val1 = inputs[in_network_offset + in_row * nbyn + in_col];
    float val2 = inputs[in_network_offset + in_row * nbyn + (in_col + 1)];
    float val3 = inputs[in_network_offset + (in_row + 1) * nbyn + in_col];
    float val4 = inputs[in_network_offset + (in_row + 1) * nbyn + (in_col + 1)];
    
    // Find max value
    float max_val = fmax(fmax(val1, val2), fmax(val3, val4));
    
    // Write result
    outputs[out_network_offset + row * out_dim + col] = max_val;
}

kernel void fc_layer(
    const device float* inputs [[buffer(0)]],
    device float* outputs [[buffer(1)]],
    const device float* network [[buffer(2)]],
    constant int& in_dim [[buffer(3)]],
    constant int& out_dim [[buffer(4)]],
    constant int& network_offset [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]
) {
    const int outIdx = position.x;
    const int batch = position.y;
    if (outIdx >= out_dim || batch >= 500) return;
    
    // Get weights and bias pointers
    const device float* weights = network + network_offset;
    const device float* bias = weights + in_dim * out_dim;
    
    // Calculate dot product
    float sum = 0.0f;
    int inputs_network_offset = batch * in_dim;
    int weight_network_offset = outIdx * in_dim;
    
    #pragma unroll(8)
    for (int i = 0; i < in_dim; ++i) {
        sum += inputs[inputs_network_offset + i] * weights[weight_network_offset + i];
    }
    
    // Add bias and apply ReLU
    sum += bias[outIdx];
    outputs[batch * out_dim + outIdx] = fmax(0.0f, sum);
}
