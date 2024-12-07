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
    // const int RTS = TILE_SIZE / WPT;

    const int idx = position.x * WPT;
    const int outNeuron = position.y;
    const int batch = position.z * BPT;
    
    // Early return for out of bounds
    if (outNeuron >= out_dim || batch >= BATCH_SIZE) return;

    // Pre-calculate offsets
    const device float* filters = networks + network_network_offset;
    const device float* biases = filters + 3 * 3 * in_dim * out_dim;
    
    // Arrays for holding intermediate results
    float sum[BPT][WPT] = {{0.0f}};

    for (int d = 0; d < BPT && (batch + d) < BATCH_SIZE; ++d) {
        const int batch_offset = (batch + d) * in_dim * nbyn * nbyn;
        
        for (int w = 0; w < WPT && (idx + w) < (nbyn * nbyn); ++w) {
            const int center_idx = idx + w;
            const int row = center_idx / nbyn;
            const int col = center_idx % nbyn;
            
            if (row >= nbyn || col >= nbyn) continue;
            
            for (int inNeuron = 0; inNeuron < in_dim; ++inNeuron) {
                const device float* current_inputs = inputs + image_network_offset + batch_offset + inNeuron * nbyn * nbyn;
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
                
                sum[d][w] += top_left * filter[0] + top * filter[1] + top_right * filter[2]
                          + left * filter[3] + center * filter[4] + right * filter[5]
                          + bottom_left * filter[6] + bottom * filter[7] + bottom_right * filter[8];
            }
        }
    }

    // Write results
    for (int d = 0; d < BPT && (batch + d) < BATCH_SIZE; ++d) {
        for (int w = 0; w < WPT && (idx + w) < (nbyn * nbyn); ++w) {
            const int center_idx = idx + w;
            int out_idx = (batch + d) * out_dim * nbyn * nbyn + outNeuron * nbyn * nbyn + center_idx;
            outputs[out_idx] = fmax(0.0f, sum[d][w] + biases[outNeuron]);  // ReLU activation
        }
    }
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
    
    if (batch >= 500 || n >= in_dim || idx >= nbyn * nbyn) return;
    
    // Calculate positions (nbyn is already output size)
    const int row = idx / nbyn;
    const int col = idx % nbyn;
    const int in_row = row * 2;
    const int in_col = col * 2;
    
    // Calculate memory offsets (input size is 2*nbyn)
    const int in_offset = batch * in_dim * (2*nbyn) * (2*nbyn) + n * (2*nbyn) * (2*nbyn);
    const int out_offset = batch * in_dim * nbyn * nbyn + n * nbyn * nbyn;

    // Get 2x2 window values from input (note the 2*nbyn stride)
    float val1 = inputs[in_offset + in_row * (2*nbyn) + in_col];
    float val2 = inputs[in_offset + in_row * (2*nbyn) + (in_col + 1)];
    float val3 = inputs[in_offset + (in_row + 1) * (2*nbyn) + in_col];
    float val4 = inputs[in_offset + (in_row + 1) * (2*nbyn) + (in_col + 1)];

    // Find max value
    float max_val = fmax(fmax(val1, val2), fmax(val3, val4));

    // Write result
    outputs[out_offset + row * nbyn + col] = max_val;
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
