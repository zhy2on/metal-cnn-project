// cnn_metal.mm
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "cnn_metal.h"
#include <time.h>
#include <string.h>

// Metal 객체들
static id<MTLDevice> device;
static id<MTLCommandQueue> commandQueue;
static id<MTLLibrary> library;
static id<MTLComputePipelineState> convPipelineState;
static id<MTLComputePipelineState> poolPipelineState;
static id<MTLComputePipelineState> fcPipelineState;

/**
* BATCH SIZE는 32의 배수로 설정해야 GPU의 병렬 처리를 최대한 활용할 수 있음.
* M1/M2 GPU에 최적화된 값으로 설정
*/

const int BATCH_SIZE = 64;

// CNN 구조 상수
const int INPUT_DIM[] = {3, 64, 64, 64, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512};
const int OUTPUT_DIM[] = {64, 64, 64, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 10};
const int NBYN[] = {32, 32, 16, 16, 16, 8, 8, 8, 8, 4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1};

void cnn_init() {
    @autoreleasepool {
        // 1. GPU 디바이스 생성
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal is not supported\n");
            return;
        }
        
        // 2. 커맨드 큐 생성
        commandQueue = [device newCommandQueue];
        
        // 3. Metal 라이브러리 로드
        NSError* error = nil;
        NSString* path = [[NSBundle mainBundle] pathForResource:@"shader" ofType:@"metallib"];
        NSURL *libraryURL = [NSURL fileURLWithPath:path];
        library = [device newLibraryWithURL:libraryURL error:&error];
        if (!library) {
            fprintf(stderr, "Failed to load Metal library: %s\n", 
                    error ? [[error localizedDescription] UTF8String] : "unknown error");
            return;
        }
        
        // 4. 파이프라인 상태 생성
        id<MTLFunction> convFunction = [library newFunctionWithName:@"convolution_kernel"];
        id<MTLFunction> poolFunction = [library newFunctionWithName:@"max_pooling_kernel"];
        id<MTLFunction> fcFunction = [library newFunctionWithName:@"fc_layer_kernel"];
        
        convPipelineState = [device newComputePipelineStateWithFunction:convFunction error:&error];
        poolPipelineState = [device newComputePipelineStateWithFunction:poolFunction error:&error];
        fcPipelineState = [device newComputePipelineStateWithFunction:fcFunction error:&error];
        
        if (!convPipelineState || !poolPipelineState || !fcPipelineState) {
            fprintf(stderr, "Failed to create pipeline states\n");
            return;
        }
    }
}

static void convolution_metal(id<MTLBuffer> input, id<MTLBuffer> output, id<MTLBuffer> network,
                            int inDim, int outDim, int nbyn, int imageOffset, int networkOffset) {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
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
        MTLSize gridSize = MTLSizeMake(nbyn, nbyn, outDim * BATCH_SIZE);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

static void max_pooling_metal(id<MTLBuffer> input, id<MTLBuffer> output, int DIM, int nbyn) {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
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
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

static void fc_layer_metal(id<MTLBuffer> input, id<MTLBuffer> output, id<MTLBuffer> network,
                          int inDim, int outDim, int networkOffset) {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
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
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

static void softmax(float* input, int N) {
    float max = input[0];
    for (int i = 1; i < N; i++) {
        if (max < input[i]) max = input[i];
    }
    
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += exp(input[i] - max);
    }
    
    for (int i = 0; i < N; i++) {
        input[i] = exp(input[i] - max) / (sum + 1e-7);
    }
}

static int find_max(float* input, int classNum) {
    int maxIndex = 0;
    float max = 0;
    for (int i = 0; i < classNum; i++) {
        if (max < input[i]) {
            max = input[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

void cnn(float* images, float* network, int* labels, float* confidences, int num_of_image) {
    time_t start, end;
    start = clock();
    
    cnn_init();
    
    @autoreleasepool {
    // Metal 버퍼 생성
    id<MTLBuffer> networkBuffer = [device newBufferWithBytes:network
                                                    length:60980520
                                                   options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> imageBuffer = [device newBufferWithBytes:images 
                                                  length:sizeof(float) * 32 * 32 * 3 * num_of_image
                                                 options:MTLResourceStorageModeShared];
    
    // 중간 결과를 저장할 버퍼들
    id<MTLBuffer> convBuffer1 = [device newBufferWithLength:sizeof(float) * 32 * 32 * 64 * BATCH_SIZE
                                                 options:MTLResourceStorageModePrivate];
    
    id<MTLBuffer> convBuffer2 = [device newBufferWithLength:sizeof(float) * 32 * 32 * 64 * BATCH_SIZE
                                                 options:MTLResourceStorageModePrivate];
    
    id<MTLBuffer> poolBuffer = [device newBufferWithLength:sizeof(float) * 16 * 16 * 64 * BATCH_SIZE
                                                options:MTLResourceStorageModePrivate];
    
    id<MTLBuffer> fcInputBuffer = [device newBufferWithLength:sizeof(float) * 512 * BATCH_SIZE
                                                   options:MTLResourceStorageModePrivate];
    
    id<MTLBuffer> fcOutputBuffer = [device newBufferWithLength:sizeof(float) * 512 * BATCH_SIZE
                                                    options:MTLResourceStorageModePrivate];
    
    float* result = (float*)malloc(sizeof(float) * 10 * BATCH_SIZE);
    id<MTLBuffer> resultBuffer = [device newBufferWithBytes:result
                                                   length:sizeof(float) * 10 * BATCH_SIZE
                                                  options:MTLResourceStorageModeShared];
    
    int network_offsets[15];
    network_offsets[0] = 3 * 3 * INPUT_DIM[0] * OUTPUT_DIM[0] + OUTPUT_DIM[0];
    network_offsets[1] = 3 * 3 * INPUT_DIM[1] * OUTPUT_DIM[1] + OUTPUT_DIM[1];
    network_offsets[2] = 3 * 3 * INPUT_DIM[3] * OUTPUT_DIM[3] + OUTPUT_DIM[3];
    network_offsets[3] = 3 * 3 * INPUT_DIM[4] * OUTPUT_DIM[4] + OUTPUT_DIM[4];
    network_offsets[4] = 3 * 3 * INPUT_DIM[6] * OUTPUT_DIM[6] + OUTPUT_DIM[6];
    network_offsets[5] = 3 * 3 * INPUT_DIM[7] * OUTPUT_DIM[7] + OUTPUT_DIM[7];
    network_offsets[6] = 3 * 3 * INPUT_DIM[8] * OUTPUT_DIM[8] + OUTPUT_DIM[8];
    network_offsets[7] = 3 * 3 * INPUT_DIM[10] * OUTPUT_DIM[10] + OUTPUT_DIM[10];
    network_offsets[8] = 3 * 3 * INPUT_DIM[11] * OUTPUT_DIM[11] + OUTPUT_DIM[11];
    network_offsets[9] = 3 * 3 * INPUT_DIM[12] * OUTPUT_DIM[12] + OUTPUT_DIM[12];
    network_offsets[10] = 3 * 3 * INPUT_DIM[14] * OUTPUT_DIM[14] + OUTPUT_DIM[14];
    network_offsets[11] = 3 * 3 * INPUT_DIM[15] * OUTPUT_DIM[15] + OUTPUT_DIM[15];
    network_offsets[12] = 3 * 3 * INPUT_DIM[16] * OUTPUT_DIM[16] + OUTPUT_DIM[16];
    network_offsets[13] = INPUT_DIM[18] * OUTPUT_DIM[18] + OUTPUT_DIM[18];
    network_offsets[14] = INPUT_DIM[19] * OUTPUT_DIM[19] + OUTPUT_DIM[19];
    
    int full_batches = num_of_image / BATCH_SIZE;
    int remaining_images = num_of_image % BATCH_SIZE;
    
    // 전체 배치 처리
    for (int i = 0; i < full_batches; ++i) {
        int image_offset = 32 * 32 * 3 * i * BATCH_SIZE;
        int network_offset = 0;
        
        // Conv block 1
        convolution_metal(imageBuffer, convBuffer1, networkBuffer, INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0], image_offset, network_offset);
        network_offset += network_offsets[0];
        
        convolution_metal(convBuffer1, convBuffer2, networkBuffer, INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1], 0, network_offset);
        network_offset += network_offsets[1];
        
        max_pooling_metal(convBuffer2, poolBuffer, INPUT_DIM[2], NBYN[2] * 2);
        
        // Conv block 2
        convolution_metal(poolBuffer, convBuffer1, networkBuffer, INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3], 0, network_offset);
        network_offset += network_offsets[2];
        
        convolution_metal(convBuffer1, convBuffer2, networkBuffer, INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4], 0, network_offset);
        network_offset += network_offsets[3];
        
        max_pooling_metal(convBuffer2, poolBuffer, INPUT_DIM[5], NBYN[5] * 2);
        
        // Conv block 3
        convolution_metal(poolBuffer, convBuffer1, networkBuffer, INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6], 0, network_offset);
        network_offset += network_offsets[4];
        
        convolution_metal(convBuffer1, convBuffer2, networkBuffer, INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7], 0, network_offset);
        network_offset += network_offsets[5];
        
        convolution_metal(convBuffer2, convBuffer1, networkBuffer, INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8], 0, network_offset);
        network_offset += network_offsets[6];
        
        max_pooling_metal(convBuffer1, poolBuffer, INPUT_DIM[9], NBYN[9] * 2);
        
        // Conv block 4
        convolution_metal(poolBuffer, convBuffer1, networkBuffer, INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10], 0, network_offset);
        network_offset += network_offsets[7];
        
        convolution_metal(convBuffer1, convBuffer2, networkBuffer, INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11], 0, network_offset);
        network_offset += network_offsets[8];
        
        convolution_metal(convBuffer2, convBuffer1, networkBuffer, INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12], 0, network_offset);
        network_offset += network_offsets[9];
        
        max_pooling_metal(convBuffer1, poolBuffer, INPUT_DIM[13], NBYN[13] * 2);
        
        // Conv block 5
        convolution_metal(poolBuffer, convBuffer1, networkBuffer, INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14], 0, network_offset);
        network_offset += network_offsets[10];
        
        convolution_metal(convBuffer1, convBuffer2, networkBuffer, INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15], 0, network_offset);
        network_offset += network_offsets[11];
        
        convolution_metal(convBuffer2, convBuffer1, networkBuffer, INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16], 0, network_offset);
        network_offset += network_offsets[12];
        
        max_pooling_metal(convBuffer1, poolBuffer, INPUT_DIM[17], NBYN[17] * 2);
        
        // Fully connected layers
        fc_layer_metal(poolBuffer, fcOutputBuffer, networkBuffer, INPUT_DIM[18], OUTPUT_DIM[18], network_offset);
        network_offset += network_offsets[13];
        
        fc_layer_metal(fcOutputBuffer, fcInputBuffer, networkBuffer, INPUT_DIM[19], OUTPUT_DIM[19], network_offset);
        network_offset += network_offsets[14];
        
        fc_layer_metal(fcInputBuffer, resultBuffer, networkBuffer, INPUT_DIM[20], OUTPUT_DIM[20], network_offset);
        
        // 결과 처리
        float* result_ptr = (float*)resultBuffer.contents;
        for (int j = 0; j < BATCH_SIZE; j++) {
            softmax(result_ptr + j * 10, 10);
            labels[i * BATCH_SIZE + j] = find_max(result_ptr + j * 10, 10);
            confidences[i * BATCH_SIZE + j] = result_ptr[j * 10 + labels[i * BATCH_SIZE + j]];
        }
    }
    
    // 남은 이미지 처리
    if (remaining_images > 0) {
        int image_offset = 32 * 32 * 3 * full_batches * BATCH_SIZE;
        int network_offset = 0;
        
        // Conv block 1
        convolution_metal(imageBuffer, convBuffer1, networkBuffer, INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0], image_offset, network_offset);
        network_offset += network_offsets[0];
        
        convolution_metal(convBuffer1, convBuffer2, networkBuffer, INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1], 0, network_offset);
        network_offset += network_offsets[1];
        
        max_pooling_metal(convBuffer2, poolBuffer, INPUT_DIM[2], NBYN[2] * 2);
        
        // Conv block 2
        convolution_metal(poolBuffer, convBuffer1, networkBuffer, INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3], 0, network_offset);
        network_offset += network_offsets[2];
        
        convolution_metal(convBuffer1, convBuffer2, networkBuffer, INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4], 0, network_offset);
        network_offset += network_offsets[3];
        
        max_pooling_metal(convBuffer2, poolBuffer, INPUT_DIM[5], NBYN[5] * 2);
        
        // Conv block 3
        convolution_metal(poolBuffer, convBuffer1, networkBuffer, INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6], 0, network_offset);
        network_offset += network_offsets[4];
        
        convolution_metal(convBuffer1, convBuffer2, networkBuffer, INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7], 0, network_offset);
        network_offset += network_offsets[5];
        
        convolution_metal(convBuffer2, convBuffer1, networkBuffer, INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8], 0, network_offset);
        network_offset += network_offsets[6];
        
        max_pooling_metal(convBuffer1, poolBuffer, INPUT_DIM[9], NBYN[9] * 2);
        
        // Conv block 4
        convolution_metal(poolBuffer, convBuffer1, networkBuffer, INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10], 0, network_offset);
        network_offset += network_offsets[7];
        
        convolution_metal(convBuffer1, convBuffer2, networkBuffer, INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11], 0, network_offset);
        network_offset += network_offsets[8];
        
        convolution_metal(convBuffer2, convBuffer1, networkBuffer, INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12], 0, network_offset);
        network_offset += network_offsets[9];
        
        max_pooling_metal(convBuffer1, poolBuffer, INPUT_DIM[13], NBYN[13] * 2);
        
        // Conv block 5
        convolution_metal(poolBuffer, convBuffer1, networkBuffer, INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14], 0, network_offset);
        network_offset += network_offsets[10];
        
        convolution_metal(convBuffer1, convBuffer2, networkBuffer, INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15], 0, network_offset);
        network_offset += network_offsets[11];
        
        convolution_metal(convBuffer2, convBuffer1, networkBuffer, INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16], 0, network_offset);
        network_offset += network_offsets[12];
        
        max_pooling_metal(convBuffer1, poolBuffer, INPUT_DIM[17], NBYN[17] * 2);
        
        // Fully connected layers
        fc_layer_metal(poolBuffer, fcOutputBuffer, networkBuffer, INPUT_DIM[18], OUTPUT_DIM[18], network_offset);
        network_offset += network_offsets[13];
        
        fc_layer_metal(fcOutputBuffer, fcInputBuffer, networkBuffer, INPUT_DIM[19], OUTPUT_DIM[19], network_offset);
        network_offset += network_offsets[14];
        
        fc_layer_metal(fcInputBuffer, resultBuffer, networkBuffer, INPUT_DIM[20], OUTPUT_DIM[20], network_offset);
        
        // 남은 이미지들의 결과 처리
        float* result_ptr = (float*)resultBuffer.contents;
        for (int j = 0; j < remaining_images; j++) {
            softmax(result_ptr + j * 10, 10);
            labels[full_batches * BATCH_SIZE + j] = find_max(result_ptr + j * 10, 10);
            confidences[full_batches * BATCH_SIZE + j] = result_ptr[j * 10 + labels[full_batches * BATCH_SIZE + j]];
        }
    }
    
    free(result);
}
    
    end = clock();
    printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLOCKS_PER_SEC);
}
