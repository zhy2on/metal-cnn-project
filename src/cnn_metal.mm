// cnn_metal.mm
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "cnn_metal.h"
#include <sys/time.h>
#include <string.h>

// Metal 컨텍스트
id<MTLDevice> device;
id<MTLCommandQueue> commandQueue;
id<MTLLibrary> library;
id<MTLComputePipelineState> convPipelineState;
id<MTLComputePipelineState> poolPipelineState;
id<MTLComputePipelineState> fcPipelineState;

// Metal 버퍼 구조체
typedef struct {
    id<MTLBuffer> networkBuffer;
    id<MTLBuffer> imageBuffer;
    id<MTLBuffer> convBuffer1;
    id<MTLBuffer> convBuffer2;
    id<MTLBuffer> poolBuffer;
    id<MTLBuffer> fcInputBuffer;
    id<MTLBuffer> fcOutputBuffer;
    id<MTLBuffer> resultBuffer;
} MetalBuffers;

// 전역 변수
static MetalBuffers buffers;

// 내부 함수 선언
static void init_metal_buffers(float* images, float* network, int num_of_image);
static void convolution_metal(id<MTLCommandBuffer> commandBuffer, 
                            id<MTLBuffer> input, id<MTLBuffer> output, id<MTLBuffer> network,
                            int inDim, int outDim, int nbyn, int imageOffset, int networkOffset);
static void max_pooling_metal(id<MTLCommandBuffer> commandBuffer, id<MTLBuffer> input, id<MTLBuffer> output, 
                            int DIM, int nbyn);
static void fc_layer_metal(id<MTLCommandBuffer> commandBuffer, id<MTLBuffer> input, id<MTLBuffer> output, 
                          id<MTLBuffer> network, int inDim, int outDim, int networkOffset);

static void process_single_batch(int image_offset, int* labels, float* confidences,
                               int batch_index, int batch_size);

static void init_metal_buffers(float* images, float* network, int num_of_image) {
    // Network weights & biases buffer
    buffers.networkBuffer = [device newBufferWithBytes:network 
                                                       length:60980520
                                                      options:MTLResourceStorageModeShared];
    
    // Input images buffer
    buffers.imageBuffer = [device newBufferWithBytes:images 
                                                     length:sizeof(float) * INPUT_SIZE * INPUT_SIZE * NUM_CHANNELS * num_of_image
                                                    options:MTLResourceStorageModeShared];
    
    // Convolution buffers
    buffers.convBuffer1 = [device newBufferWithLength:sizeof(float) * 32 * 32 * 64 * BATCH_SIZE
                                                     options:MTLResourceStorageModePrivate];
    buffers.convBuffer2 = [device newBufferWithLength:sizeof(float) * 32 * 32 * 64 * BATCH_SIZE
                                                     options:MTLResourceStorageModePrivate];
    
    // Pooling buffer
    buffers.poolBuffer = [device newBufferWithLength:sizeof(float) * 16 * 16 * 64 * BATCH_SIZE
                                                    options:MTLResourceStorageModePrivate];
    
    // FC layer buffers
    buffers.fcInputBuffer = [device newBufferWithLength:sizeof(float) * 512 * BATCH_SIZE
                                                       options:MTLResourceStorageModePrivate];
    buffers.fcOutputBuffer = [device newBufferWithLength:sizeof(float) * 512 * BATCH_SIZE
                                                        options:MTLResourceStorageModePrivate];
    
    // Result buffer
    buffers.resultBuffer = [device newBufferWithLength:sizeof(float) * NUM_CLASSES * BATCH_SIZE
                                                      options:MTLResourceStorageModeShared];
}

static void convolution_metal(id<MTLCommandBuffer> commandBuffer, 
                            id<MTLBuffer> input, id<MTLBuffer> output, id<MTLBuffer> network,
                            int inDim, int outDim, int nbyn, int imageOffset, int networkOffset) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    int params[] = {inDim, outDim, nbyn, imageOffset, networkOffset};
    id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:params 
                                                   length:sizeof(params) 
                                                  options:MTLResourceStorageModeShared];
    
    [encoder setComputePipelineState:convPipelineState];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:network offset:0 atIndex:2];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:3];
    
    MTLSize threadGroupSize = MTLSizeMake(8, 8, 1);
    MTLSize gridSize = MTLSizeMake(nbyn * nbyn, outDim, BATCH_SIZE);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
}

static void max_pooling_metal(id<MTLCommandBuffer> commandBuffer,
                            id<MTLBuffer> input, id<MTLBuffer> output, 
                            int DIM, int nbyn) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    int params[] = {DIM, nbyn/2};
    id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:params 
                                                   length:sizeof(params) 
                                                  options:MTLResourceStorageModeShared];
    
    [encoder setComputePipelineState:poolPipelineState];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
    
    MTLSize threadGroupSize = MTLSizeMake(8, 8, 1);
    MTLSize gridSize = MTLSizeMake(DIM, nbyn/2 * nbyn/2, BATCH_SIZE);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
}

static void fc_layer_metal(id<MTLCommandBuffer> commandBuffer,
                          id<MTLBuffer> input, id<MTLBuffer> output, id<MTLBuffer> network,
                          int inDim, int outDim, int networkOffset) {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    int params[] = {inDim, outDim, networkOffset};
    id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:params 
                                                   length:sizeof(params) 
                                                  options:MTLResourceStorageModeShared];
    
    [encoder setComputePipelineState:fcPipelineState];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBuffer:network offset:0 atIndex:2];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:3];
    
    MTLSize threadGroupSize = MTLSizeMake(MIN((int)fcPipelineState.maxTotalThreadsPerThreadgroup, outDim), 1, 1);
    MTLSize gridSize = MTLSizeMake(outDim, BATCH_SIZE, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
}

static void softmax(float* input, int N) {
    float max = input[0];
    for (int i = 1; i < N; ++i) {
        if (max < input[i]) max = input[i];
    }
    
    float sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += exp(input[i] - max);
    }
    
    for (int i = 0; i < N; ++i) {
        input[i] = exp(input[i] - max) / (sum + 1e-7);
    }
}

static int find_max(float* input, int classNum) {
    int maxIndex = 0;
    float max = 0;
    for (int i = 0; i < classNum; ++i) {
        if (max < input[i]) {
            max = input[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

static void process_single_batch(int image_offset, int* labels, float* confidences,
                               int batch_index, int batch_size) {
	id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    int network_offset = 0;
    
    // Conv block 1
    convolution_metal(commandBuffer, buffers.imageBuffer, buffers.convBuffer1, buffers.networkBuffer, 
                    INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0], image_offset, network_offset);
    network_offset += NETWORK_OFFSETS[0];
    
    convolution_metal(commandBuffer, buffers.convBuffer1, buffers.convBuffer2, buffers.networkBuffer, 
                    INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1], 0, network_offset);
    network_offset += NETWORK_OFFSETS[1];
    
    max_pooling_metal(commandBuffer, buffers.convBuffer2, buffers.poolBuffer, INPUT_DIM[2], NBYN[2] * 2);
    
    // Conv block 2
    convolution_metal(commandBuffer, buffers.poolBuffer, buffers.convBuffer1, buffers.networkBuffer,
                    INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3], 0, network_offset);
    network_offset += NETWORK_OFFSETS[2];
    
    convolution_metal(commandBuffer, buffers.convBuffer1, buffers.convBuffer2, buffers.networkBuffer,
                    INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4], 0, network_offset);
    network_offset += NETWORK_OFFSETS[3];
    
    max_pooling_metal(commandBuffer, buffers.convBuffer2, buffers.poolBuffer, INPUT_DIM[5], NBYN[5] * 2);
    
    // Conv block 3
    convolution_metal(commandBuffer, buffers.poolBuffer, buffers.convBuffer1, buffers.networkBuffer,
                    INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6], 0, network_offset);
    network_offset += NETWORK_OFFSETS[4];
    
    convolution_metal(commandBuffer, buffers.convBuffer1, buffers.convBuffer2, buffers.networkBuffer,
                    INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7], 0, network_offset);
    network_offset += NETWORK_OFFSETS[5];
    
    convolution_metal(commandBuffer, buffers.convBuffer2, buffers.convBuffer1, buffers.networkBuffer,
                    INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8], 0, network_offset);
    network_offset += NETWORK_OFFSETS[6];
    
    max_pooling_metal(commandBuffer, buffers.convBuffer1, buffers.poolBuffer, INPUT_DIM[9], NBYN[9] * 2);
    
    // Conv block 4
    convolution_metal(commandBuffer, buffers.poolBuffer, buffers.convBuffer1, buffers.networkBuffer,
                        INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10], 0, network_offset);
    network_offset += NETWORK_OFFSETS[7];
    
    convolution_metal(commandBuffer, buffers.convBuffer1, buffers.convBuffer2, buffers.networkBuffer,
                        INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11], 0, network_offset);
    network_offset += NETWORK_OFFSETS[8];
    
    convolution_metal(commandBuffer, buffers.convBuffer2, buffers.convBuffer1, buffers.networkBuffer,
                        INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12], 0, network_offset);
    network_offset += NETWORK_OFFSETS[9];
    
    max_pooling_metal(commandBuffer, buffers.convBuffer1, buffers.poolBuffer, INPUT_DIM[13], NBYN[13] * 2);
    
    // Conv block 5
    convolution_metal(commandBuffer, buffers.poolBuffer, buffers.convBuffer1, buffers.networkBuffer,
                        INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14], 0, network_offset);
    network_offset += NETWORK_OFFSETS[10];
    
    convolution_metal(commandBuffer, buffers.convBuffer1, buffers.convBuffer2, buffers.networkBuffer,
                        INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15], 0, network_offset);
    network_offset += NETWORK_OFFSETS[11];
    
    convolution_metal(commandBuffer, buffers.convBuffer2, buffers.convBuffer1, buffers.networkBuffer,
                        INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16], 0, network_offset);
    network_offset += NETWORK_OFFSETS[12];
    
    max_pooling_metal(commandBuffer, buffers.convBuffer1, buffers.poolBuffer, INPUT_DIM[17], NBYN[17] * 2);
    
    // Fully connected layers
    fc_layer_metal(commandBuffer, buffers.poolBuffer, buffers.fcOutputBuffer, buffers.networkBuffer,
                        INPUT_DIM[18], OUTPUT_DIM[18], network_offset);
    network_offset += NETWORK_OFFSETS[13];
    
    fc_layer_metal(commandBuffer, buffers.fcOutputBuffer, buffers.fcInputBuffer, buffers.networkBuffer,
                        INPUT_DIM[19], OUTPUT_DIM[19], network_offset);
    network_offset += NETWORK_OFFSETS[14];
    
    fc_layer_metal(commandBuffer, buffers.fcInputBuffer, buffers.resultBuffer, buffers.networkBuffer,
                        INPUT_DIM[20], OUTPUT_DIM[20], network_offset);
    
	[commandBuffer commit];
	[commandBuffer waitUntilCompleted];

    // 결과 처리
    float* result_ptr = (float*)buffers.resultBuffer.contents;
    for (int j = 0; j < batch_size; ++j) {
        softmax(result_ptr + j * 10, 10);
        int index = batch_index * BATCH_SIZE + j;
        labels[index] = find_max(result_ptr + j * 10, 10);
        confidences[index] = result_ptr[j * 10 + labels[index]];
    }
}

void cnn_init(void) {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal is not supported\n");
            free(device);
            return;
        }
        
        commandQueue = [device newCommandQueue];
        
        NSError* error = nil;
        NSString* path = [[NSBundle mainBundle] pathForResource:@"shader" ofType:@"metallib"];
        NSURL *libraryURL = [NSURL fileURLWithPath:path];
        library = [device newLibraryWithURL:libraryURL error:&error];
        
        if (!library) {
            fprintf(stderr, "Failed to load Metal library: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown error");
            free(device);
            return;
        }
        
        id<MTLFunction> convFunction = [library newFunctionWithName:@"convolution_kernel"];
        id<MTLFunction> poolFunction = [library newFunctionWithName:@"max_pooling_kernel"];
        id<MTLFunction> fcFunction = [library newFunctionWithName:@"fc_layer_kernel"];
        
        convPipelineState = [device newComputePipelineStateWithFunction:convFunction error:&error];
        poolPipelineState = [device newComputePipelineStateWithFunction:poolFunction error:&error];
        fcPipelineState = [device newComputePipelineStateWithFunction:fcFunction error:&error];
    }
}

void cnn(float* images, float* network, int* labels, float* confidences, int num_of_image) {
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    cnn_init();
    
    @autoreleasepool {
        // Metal 버퍼 초기화
        init_metal_buffers(images, network, num_of_image);
    
        int full_batches = num_of_image / BATCH_SIZE;
        int remaining_images = num_of_image % BATCH_SIZE;
    
        // 전체 배치 처리
        for (int i = 0; i < full_batches; ++i) {
            process_single_batch(INPUT_SIZE * INPUT_SIZE * NUM_CHANNELS * i * BATCH_SIZE,
                               labels, confidences, i, BATCH_SIZE);
        }
    
        // 남은 이미지 처리
        if (remaining_images > 0) {
            process_single_batch(INPUT_SIZE * INPUT_SIZE * NUM_CHANNELS * full_batches * BATCH_SIZE,
                               labels, confidences, full_batches, remaining_images);
        }
    }
    
    gettimeofday(&end_time, NULL);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    printf("Elapsed time: %.6f seconds\n", elapsed_time);
}
