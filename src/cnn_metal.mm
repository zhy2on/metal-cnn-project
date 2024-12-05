#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "cnn_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Metal 관련 전역 변수
id<MTLDevice> device;
id<MTLCommandQueue> commandQueue[2];
id<MTLLibrary> defaultLibrary;
id<MTLFunction> convolutionFunction;
id<MTLFunction> maxPoolingFunction;
id<MTLFunction> fcLayerFunction;
id<MTLComputePipelineState> convolutionPipeline;
id<MTLComputePipelineState> maxPoolingPipeline;
id<MTLComputePipelineState> fcLayerPipeline;

// 메모리 버퍼 전역 변수
id<MTLBuffer> bufferImages;
id<MTLBuffer> bufferNetwork;
id<MTLBuffer> bufferConvolutionInput[2];
id<MTLBuffer> bufferConvolutionOutput[2];
id<MTLBuffer> bufferPoolingOutput[2];
id<MTLBuffer> bufferFcInput[2];
id<MTLBuffer> bufferFcOutput[2];

const int BATCH_SIZE = 500;

const int INPUT_DIM[] = {3,	  64,  64,

						 64,  128, 128,

						 128, 256, 256, 256,

						 256, 512, 512, 512,

						 512, 512, 512, 512,

						 512, 512, 512};

const int OUTPUT_DIM[] = {64,  64,	64,

						  128, 128, 128,

						  256, 256, 256, 256,

						  512, 512, 512, 512,

						  512, 512, 512, 512,

						  512, 512, 10};

const int NBYN[] = {32, 32, 16,

					16, 16, 8,

					8,	8,	8,	4,

					4,	4,	4,	2,

					2,	2,	2,	1,

					1,	1,	1};

void convolution_metal(id<MTLCommandBuffer> commandBuffer, id<MTLBuffer> inputs, id<MTLBuffer> outputs, 
                      int inDim, int outDim, int nbyn, int image_offset, int network_offset) {
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    [computeEncoder setComputePipelineState:convolutionPipeline];
    [computeEncoder setBuffer:inputs offset:0 atIndex:0];
    [computeEncoder setBuffer:outputs offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferNetwork offset:0 atIndex:2];
    
    [computeEncoder setBytes:&inDim length:sizeof(int) atIndex:3];
    [computeEncoder setBytes:&outDim length:sizeof(int) atIndex:4];
    [computeEncoder setBytes:&nbyn length:sizeof(int) atIndex:5];
    [computeEncoder setBytes:&image_offset length:sizeof(int) atIndex:6];
    [computeEncoder setBytes:&network_offset length:sizeof(int) atIndex:7];
    
    MTLSize gridSize = MTLSizeMake(nbyn * nbyn, outDim, BATCH_SIZE);
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
    
    [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];
}

void max_pooling_metal(id<MTLCommandBuffer> commandBuffer, id<MTLBuffer> input, id<MTLBuffer> output, 
                      int DIM, int nbyn) {
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    [computeEncoder setComputePipelineState:maxPoolingPipeline];
    [computeEncoder setBuffer:input offset:0 atIndex:0];
    [computeEncoder setBuffer:output offset:0 atIndex:1];
    [computeEncoder setBytes:&DIM length:sizeof(int) atIndex:2];
    [computeEncoder setBytes:&nbyn length:sizeof(int) atIndex:3];
    
    MTLSize gridSize = MTLSizeMake(DIM, nbyn / 2 * nbyn / 2, BATCH_SIZE);
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
    
    [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];
}

void fc_layer_metal(id<MTLCommandBuffer> commandBuffer, id<MTLBuffer> input, id<MTLBuffer> output, 
                   int inDim, int outDim, int network_offset) {
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    [computeEncoder setComputePipelineState:fcLayerPipeline];
    [computeEncoder setBuffer:input offset:0 atIndex:0];
    [computeEncoder setBuffer:output offset:0 atIndex:1];
    [computeEncoder setBuffer:bufferNetwork offset:0 atIndex:2];
    [computeEncoder setBytes:&inDim length:sizeof(int) atIndex:3];
    [computeEncoder setBytes:&outDim length:sizeof(int) atIndex:4];
    [computeEncoder setBytes:&network_offset length:sizeof(int) atIndex:5];
    
    MTLSize gridSize = MTLSizeMake(outDim, BATCH_SIZE, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    
    [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];
}

void softmax(float* output, int N) {
    int i;
    float max = output[0];
    for (i = 1; i < N; i++) {
        max = (output[i] > max) ? output[i] : max;
    }
    float sum = 0;
    for (i = 0; i < N; i++) {
        sum += exp(output[i] - max);
    }
    for (i = 0; i < N; i++) {
        output[i] = exp(output[i] - max) / sum;
    }
}

int find_max(float* fc, int N) {
    int i;
    int maxid = 0;
    float maxval = 0;
    for (i = 0; i < N; i++) {
        if (maxval < fc[i]) {
            maxval = fc[i];
            maxid = i;
        }
    }
    return maxid;
}

void cnn_init(float* images, float* network, int* labels, float* confidences, int num_images) {
    // Metal 디바이스 생성
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("Failed to create Metal device\n");
        exit(1);
    }
    
    // 두 개의 커맨드 큐 생성
    for(int i = 0; i < 2; i++) {
        commandQueue[i] = [device newCommandQueue];
        if (!commandQueue[i]) {
            printf("Failed to create command queue %d\n", i);
            exit(1);
        }
    }
    
    // Metal 라이브러리 로드
    NSError* error = nil;
    NSString* path = [[NSBundle mainBundle] pathForResource:@"shader" ofType:@"metallib"];
    NSURL *libraryURL = [NSURL fileURLWithPath:path];
    defaultLibrary = [device newLibraryWithURL:libraryURL error:&error];
    
    if (!defaultLibrary) {
        printf("Failed to load Metal library: %s\n",
                error ? [[error localizedDescription] UTF8String] : "unknown error");
        exit(1);
    }
    
    // 커널 함수 로드
    convolutionFunction = [defaultLibrary newFunctionWithName:@"convolution"];
    maxPoolingFunction = [defaultLibrary newFunctionWithName:@"max_pooling"];
    fcLayerFunction = [defaultLibrary newFunctionWithName:@"fc_layer"];
    
    // 파이프라인 상태 객체 생성
    convolutionPipeline = [device newComputePipelineStateWithFunction:convolutionFunction error:&error];
    maxPoolingPipeline = [device newComputePipelineStateWithFunction:maxPoolingFunction error:&error];
    fcLayerPipeline = [device newComputePipelineStateWithFunction:fcLayerFunction error:&error];
    
    // 버퍼 생성
    bufferImages = [device newBufferWithBytes:images 
                                     length:sizeof(float) * 32 * 32 * 3 * num_images 
                                    options:MTLResourceStorageModeShared];
    
    bufferNetwork = [device newBufferWithBytes:network 
                                      length:60980520 
                                     options:MTLResourceStorageModeShared];
    
    for (int i = 0; i < 2; i++) {
        bufferConvolutionInput[i] = [device newBufferWithLength:sizeof(float) * 32 * 32 * 64 * BATCH_SIZE 
                                                      options:MTLResourceStorageModeShared];
        
        bufferConvolutionOutput[i] = [device newBufferWithLength:sizeof(float) * 32 * 32 * 64 * BATCH_SIZE 
                                                       options:MTLResourceStorageModeShared];
        
        bufferPoolingOutput[i] = [device newBufferWithLength:sizeof(float) * 16 * 16 * 64 * BATCH_SIZE 
                                                   options:MTLResourceStorageModeShared];
        
        bufferFcInput[i] = [device newBufferWithLength:sizeof(float) * 512 * BATCH_SIZE 
                                             options:MTLResourceStorageModeShared];
        
        bufferFcOutput[i] = [device newBufferWithLength:sizeof(float) * 512 * BATCH_SIZE 
                                              options:MTLResourceStorageModeShared];
    }
}

void initialize_network_offsets(int* offsets) {
    // Convolution layers (0-12)
    offsets[0] = 3 * 3 * INPUT_DIM[0] * OUTPUT_DIM[0] + OUTPUT_DIM[0];
    offsets[1] = 3 * 3 * INPUT_DIM[1] * OUTPUT_DIM[1] + OUTPUT_DIM[1];
    offsets[2] = 3 * 3 * INPUT_DIM[3] * OUTPUT_DIM[3] + OUTPUT_DIM[3];
    offsets[3] = 3 * 3 * INPUT_DIM[4] * OUTPUT_DIM[4] + OUTPUT_DIM[4];
    offsets[4] = 3 * 3 * INPUT_DIM[6] * OUTPUT_DIM[6] + OUTPUT_DIM[6];
    offsets[5] = 3 * 3 * INPUT_DIM[7] * OUTPUT_DIM[7] + OUTPUT_DIM[7];
    offsets[6] = 3 * 3 * INPUT_DIM[8] * OUTPUT_DIM[8] + OUTPUT_DIM[8];
    offsets[7] = 3 * 3 * INPUT_DIM[10] * OUTPUT_DIM[10] + OUTPUT_DIM[10];
    offsets[8] = 3 * 3 * INPUT_DIM[11] * OUTPUT_DIM[11] + OUTPUT_DIM[11];
    offsets[9] = 3 * 3 * INPUT_DIM[12] * OUTPUT_DIM[12] + OUTPUT_DIM[12];
    offsets[10] = 3 * 3 * INPUT_DIM[14] * OUTPUT_DIM[14] + OUTPUT_DIM[14];
    offsets[11] = 3 * 3 * INPUT_DIM[15] * OUTPUT_DIM[15] + OUTPUT_DIM[15];
    offsets[12] = 3 * 3 * INPUT_DIM[16] * OUTPUT_DIM[16] + OUTPUT_DIM[16];

    // FC layers (13-14)
    offsets[13] = INPUT_DIM[18] * OUTPUT_DIM[18] + OUTPUT_DIM[18];
    offsets[14] = INPUT_DIM[19] * OUTPUT_DIM[19] + OUTPUT_DIM[19];
}

id<MTLCommandBuffer> process_single_batch(const int* network_offsets, int batch_start_idx, 
                                        int current_batch_size, int buffer_idx, int queue_idx) {
    int image_offset = 32 * 32 * 3 * batch_start_idx;
    int network_offset = 0;

    // 해당 큐를 사용
    id<MTLCommandBuffer> commandBuffer = [commandQueue[queue_idx] commandBuffer];

    // Convolution layer 1
    convolution_metal(commandBuffer, bufferImages, bufferConvolutionOutput[buffer_idx], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0], 
                     image_offset, network_offset);
    image_offset = 0;
    network_offset += network_offsets[0];

    // Convolution layer 2
    convolution_metal(commandBuffer, bufferConvolutionOutput[buffer_idx], bufferConvolutionInput[buffer_idx], INPUT_DIM[1], OUTPUT_DIM[1], 
                     NBYN[1], image_offset, network_offset);
    network_offset += network_offsets[1];
    max_pooling_metal(commandBuffer, bufferConvolutionInput[buffer_idx], bufferPoolingOutput[buffer_idx], INPUT_DIM[2], NBYN[2] * 2);

    // Convolution block 1
    convolution_metal(commandBuffer, bufferPoolingOutput[buffer_idx], bufferConvolutionOutput[buffer_idx], INPUT_DIM[3], OUTPUT_DIM[3], 
                     NBYN[3], image_offset, network_offset);
    network_offset += network_offsets[2];
    convolution_metal(commandBuffer, bufferConvolutionOutput[buffer_idx], bufferConvolutionInput[buffer_idx], INPUT_DIM[4], OUTPUT_DIM[4], 
                     NBYN[4], image_offset, network_offset);
    network_offset += network_offsets[3];
    max_pooling_metal(commandBuffer, bufferConvolutionInput[buffer_idx], bufferPoolingOutput[buffer_idx], INPUT_DIM[5], NBYN[5] * 2);

    // Convolution block 2
    convolution_metal(commandBuffer, bufferPoolingOutput[buffer_idx], bufferConvolutionOutput[buffer_idx], INPUT_DIM[6], OUTPUT_DIM[6], 
                     NBYN[6], image_offset, network_offset);
    network_offset += network_offsets[4];
    convolution_metal(commandBuffer, bufferConvolutionOutput[buffer_idx], bufferConvolutionInput[buffer_idx], INPUT_DIM[7], OUTPUT_DIM[7], 
                     NBYN[7], image_offset, network_offset);
    network_offset += network_offsets[5];
    convolution_metal(commandBuffer, bufferConvolutionInput[buffer_idx], bufferConvolutionOutput[buffer_idx], INPUT_DIM[8], OUTPUT_DIM[8], 
                     NBYN[8], image_offset, network_offset);
    network_offset += network_offsets[6];
    max_pooling_metal(commandBuffer, bufferConvolutionOutput[buffer_idx], bufferPoolingOutput[buffer_idx], INPUT_DIM[9], NBYN[9] * 2);

    // Convolution block 3
    convolution_metal(commandBuffer, bufferPoolingOutput[buffer_idx], bufferConvolutionOutput[buffer_idx], INPUT_DIM[10], OUTPUT_DIM[10], 
                     NBYN[10], image_offset, network_offset);
    network_offset += network_offsets[7];
    convolution_metal(commandBuffer, bufferConvolutionOutput[buffer_idx], bufferConvolutionInput[buffer_idx], INPUT_DIM[11], OUTPUT_DIM[11], 
                     NBYN[11], image_offset, network_offset);
    network_offset += network_offsets[8];
    convolution_metal(commandBuffer, bufferConvolutionInput[buffer_idx], bufferConvolutionOutput[buffer_idx], INPUT_DIM[12], OUTPUT_DIM[12], 
                     NBYN[12], image_offset, network_offset);
    network_offset += network_offsets[9];
    max_pooling_metal(commandBuffer, bufferConvolutionOutput[buffer_idx], bufferPoolingOutput[buffer_idx], INPUT_DIM[13], NBYN[13] * 2);

    // Convolution block 4
    convolution_metal(commandBuffer, bufferPoolingOutput[buffer_idx], bufferConvolutionOutput[buffer_idx], INPUT_DIM[14], OUTPUT_DIM[14], 
                     NBYN[14], image_offset, network_offset);
    network_offset += network_offsets[10];
    convolution_metal(commandBuffer, bufferConvolutionOutput[buffer_idx], bufferConvolutionInput[buffer_idx], INPUT_DIM[15], OUTPUT_DIM[15], 
                     NBYN[15], image_offset, network_offset);
    network_offset += network_offsets[11];
    convolution_metal(commandBuffer, bufferConvolutionInput[buffer_idx], bufferConvolutionOutput[buffer_idx], INPUT_DIM[16], OUTPUT_DIM[16], 
                     NBYN[16], image_offset, network_offset);
    network_offset += network_offsets[12];
    max_pooling_metal(commandBuffer, bufferConvolutionOutput[buffer_idx], bufferPoolingOutput[buffer_idx], INPUT_DIM[17], NBYN[17] * 2);

    // FC layers
    fc_layer_metal(commandBuffer, bufferPoolingOutput[buffer_idx], bufferFcOutput[buffer_idx], INPUT_DIM[18], OUTPUT_DIM[18], network_offset);
    network_offset += network_offsets[13];
    fc_layer_metal(commandBuffer, bufferFcOutput[buffer_idx], bufferFcInput[buffer_idx], INPUT_DIM[19], OUTPUT_DIM[19], network_offset);
    network_offset += network_offsets[14];
    fc_layer_metal(commandBuffer, bufferFcInput[buffer_idx], bufferFcOutput[buffer_idx], INPUT_DIM[20], OUTPUT_DIM[20], network_offset);

    [commandBuffer commit];
    return commandBuffer;
}

void process_batch_result(id<MTLCommandBuffer> commandBuffer, float* result, int buffer_idx, int batch_size) {
    // 이전 연산이 완료될 때까지 대기
    [commandBuffer waitUntilCompleted];
    
    // 결과 복사
    float* fc_output_ptr = (float*)[bufferFcOutput[buffer_idx] contents];
    memcpy(result, fc_output_ptr, sizeof(float) * 10 * batch_size);
}

void process_results(float* result, int* labels, float* confidences, 
                    int batch_start_idx, int current_batch_size) {
    for (int j = 0; j < current_batch_size; j++) {
        softmax(result + j * 10, 10);
        labels[batch_start_idx + j] = find_max(result + j * 10, 10);
        confidences[batch_start_idx + j] = result[j * 10 + labels[batch_start_idx + j]];
    }
}

void cnn(float* images, float* network, int* labels, float* confidences, int num_images) {
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // CNN 초기화
    cnn_init(images, network, labels, confidences, num_images);

    // 네트워크 오프셋 초기화
    int network_offsets[15];
    initialize_network_offsets(network_offsets);

    // 결과를 저장할 버퍼
    float* result[2];
    result[0] = (float*)malloc(sizeof(float) * 10 * BATCH_SIZE);
    result[1] = (float*)malloc(sizeof(float) * 10 * BATCH_SIZE);

    // 배치 계산
    int full_batches = num_images / BATCH_SIZE;

    // 시간 측정 시작
    clock_t start = clock();

    // 첫 번째 배치 처리
    id<MTLCommandBuffer> prevCommandBuffer = process_single_batch(network_offsets, 
                                                                0, BATCH_SIZE, 0, 0);

    // 나머지 배치 처리
    for (int i = 1; i < full_batches; i++) {
        // 다음 배치 계산을 다른 큐에서 시작
        id<MTLCommandBuffer> currentCommandBuffer = process_single_batch(network_offsets, 
                                                                       i * BATCH_SIZE, 
                                                                       BATCH_SIZE, 
                                                                       i % 2,
                                                                       i % 2); // 큐 번갈아가며 사용
        
        // 이전 배치 결과 처리
        process_batch_result(prevCommandBuffer, result[(i-1) % 2], (i-1) % 2, BATCH_SIZE);
        process_results(result[(i-1) % 2], labels, confidences, 
                       (i-1) * BATCH_SIZE, BATCH_SIZE);
        
        prevCommandBuffer = currentCommandBuffer;
    }

    // 마지막 정규 배치 처리
    process_batch_result(prevCommandBuffer, result[(full_batches-1) % 2], 
                        (full_batches-1) % 2, BATCH_SIZE);
    process_results(result[(full_batches-1) % 2], labels, confidences, 
                   (full_batches-1) * BATCH_SIZE, BATCH_SIZE);

    // 시간 측정 종료
    gettimeofday(&end_time, NULL);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + 
                         (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    printf("Elapsed time: %.6f seconds\n", elapsed_time);

    // 메모리 해제
    free(result[0]);
    free(result[1]);

    [bufferImages release];
    [bufferNetwork release];
    for (int i = 0; i < 2; ++i) {
        [bufferConvolutionInput[i] release];
        [bufferConvolutionOutput[i] release];
        [bufferPoolingOutput[i] release];
        [bufferFcInput[i] release];
        [bufferFcOutput[i] release];
    }
    [convolutionPipeline release];
    [maxPoolingPipeline release];
    [fcLayerPipeline release];
    [convolutionFunction release];
    [maxPoolingFunction release];
    [fcLayerFunction release];
    [defaultLibrary release];
    for(int i = 0; i < 2; i++) {
        [commandQueue[i] release];
    }
    [device release];
}
