#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "cnn_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Metal 관련 전역 변수
id<MTLDevice> device;
id<MTLCommandQueue> queue;
id<MTLLibrary> library;
id<MTLFunction> f_conv;
id<MTLFunction> f_pool;
id<MTLFunction> f_fc;
id<MTLComputePipelineState> p_conv;
id<MTLComputePipelineState> p_pool;
id<MTLComputePipelineState> p_fc;

// 메모리 버퍼 전역 변수
id<MTLBuffer> b_imgs;
id<MTLBuffer> b_net;
id<MTLBuffer> b_conv_in;
id<MTLBuffer> b_conv_out;
id<MTLBuffer> b_pool_out;
id<MTLBuffer> b_fc_in;
id<MTLBuffer> b_fc_out;

const int BATCH_SIZE = 500;

const int INPUT_in_dim[] = {3,	  64,  64,

						 64,  128, 128,

						 128, 256, 256, 256,

						 256, 512, 512, 512,

						 512, 512, 512, 512,

						 512, 512, 512};

const int OUTPUT_in_dim[] = {64,  64,	64,

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

void setupMetal() {
    // Metal 디바이스 생성
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        printf("Failed to create Metal device\n");
        exit(1);
    }
    
    // 커맨드 큐 생성
    queue = [device newCommandQueue];
    if (!queue) {
        printf("Failed to create command queue\n");
        exit(1);
    }
    
    // Metal 라이브러리 로드
    NSError* error = nil;
    library = [device newDefaultLibrary];
    if (!library) {
        printf("Failed to load default library\n");
        exit(1);
    }
    
    // 커널 함수 로드
    f_conv = [library newFunctionWithName:@"convolution"];
    f_pool = [library newFunctionWithName:@"max_pooling"];
    f_fc = [library newFunctionWithName:@"fc_layer"];
    
    // 파이프라인 상태 객체 생성
    p_conv = [device newComputePipelineStateWithFunction:f_conv error:&error];
    p_pool = [device newComputePipelineStateWithFunction:f_pool error:&error];
    p_fc = [device newComputePipelineStateWithFunction:f_fc error:&error];
}

void convolution_metal(id<MTLBuffer> inputs, id<MTLBuffer> outputs, int in_dim, int out_dim, int nbyn, 
                      int image_offset, int network_offset) {
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    [computeEncoder setComputePipelineState:p_conv];
    [computeEncoder setBuffer:inputs offset:0 atIndex:0];
    [computeEncoder setBuffer:outputs offset:0 atIndex:1];
    [computeEncoder setBuffer:b_net offset:0 atIndex:2];
    
    // Parameters
    [computeEncoder setBytes:&in_dim length:sizeof(int) atIndex:3];
    [computeEncoder setBytes:&out_dim length:sizeof(int) atIndex:4];
    [computeEncoder setBytes:&nbyn length:sizeof(int) atIndex:5];
    [computeEncoder setBytes:&image_offset length:sizeof(int) atIndex:6];
    [computeEncoder setBytes:&network_offset length:sizeof(int) atIndex:7];
    
    MTLSize gridSize = MTLSizeMake(nbyn * nbyn, out_dim, BATCH_SIZE);
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
    
    [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void max_pooling_metal(id<MTLBuffer> inputs, id<MTLBuffer> outputs, int in_dim, int nbyn) {
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    [computeEncoder setComputePipelineState:p_pool];
    [computeEncoder setBuffer:inputs offset:0 atIndex:0];
    [computeEncoder setBuffer:outputs offset:0 atIndex:1];
    [computeEncoder setBytes:&in_dim length:sizeof(int) atIndex:2];
    [computeEncoder setBytes:&nbyn length:sizeof(int) atIndex:3];
    
    MTLSize gridSize = MTLSizeMake(in_dim, nbyn / 2 * nbyn / 2, BATCH_SIZE);
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
    
    [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void fc_layer_metal(id<MTLBuffer> inputs, id<MTLBuffer> outputs, int in_dim, int out_dim, int network_offset) {
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    
    [computeEncoder setComputePipelineState:p_fc];
    [computeEncoder setBuffer:inputs offset:0 atIndex:0];
    [computeEncoder setBuffer:outputs offset:0 atIndex:1];
    [computeEncoder setBuffer:b_net offset:0 atIndex:2];
    [computeEncoder setBytes:&in_dim length:sizeof(int) atIndex:3];
    [computeEncoder setBytes:&out_dim length:sizeof(int) atIndex:4];
    [computeEncoder setBytes:&network_offset length:sizeof(int) atIndex:5];
    
    MTLSize gridSize = MTLSizeMake(out_dim, BATCH_SIZE, 1);
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    
    [computeEncoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
    [computeEncoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}

void softmax(float* outputs, int N) {
    int i;
    float max = outputs[0];
    for (i = 1; i < N; i++) {
        max = (outputs[i] > max) ? outputs[i] : max;
    }
    float sum = 0;
    for (i = 0; i < N; i++) {
        sum += exp(outputs[i] - max);
    }
    for (i = 0; i < N; i++) {
        outputs[i] = exp(outputs[i] - max) / sum;
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
    
    // 커맨드 큐 생성
    queue = [device newCommandQueue];
    if (!queue) {
        printf("Failed to create command queue\n");
        exit(1);
    }
    
    // Metal 라이브러리 로드
    NSError* error = nil;
    NSString* path = [[NSBundle mainBundle] pathForResource:@"shader" ofType:@"metallib"];
    NSURL *libraryURL = [NSURL fileURLWithPath:path];
    library = [device newLibraryWithURL:libraryURL error:&error];
    
    if (!library) {
        printf("Failed to load Metal library: %s\n",
                error ? [[error localizedDescription] UTF8String] : "unknown error");
        exit(1);
    }
    
    // 커널 함수 로드
    f_conv = [library newFunctionWithName:@"convolution"];
    f_pool = [library newFunctionWithName:@"max_pooling"];
    f_fc = [library newFunctionWithName:@"fc_layer"];
    
    // 파이프라인 상태 객체 생성
    p_conv = [device newComputePipelineStateWithFunction:f_conv error:&error];
    p_pool = [device newComputePipelineStateWithFunction:f_pool error:&error];
    p_fc = [device newComputePipelineStateWithFunction:f_fc error:&error];
    
    // 버퍼 생성
    b_imgs = [device newBufferWithBytes:images 
                                     length:sizeof(float) * 32 * 32 * 3 * num_images 
                                    options:MTLResourceStorageModeShared];
    
    b_net = [device newBufferWithBytes:network 
                                      length:60980520 
                                     options:MTLResourceStorageModeShared];
    
    b_conv_in = [device newBufferWithLength:sizeof(float) * 32 * 32 * 64 * BATCH_SIZE 
                                               options:MTLResourceStorageModeShared];
    
    b_conv_out = [device newBufferWithLength:sizeof(float) * 32 * 32 * 64 * BATCH_SIZE 
                                                options:MTLResourceStorageModeShared];
    
    b_pool_out = [device newBufferWithLength:sizeof(float) * 16 * 16 * 64 * BATCH_SIZE 
                                            options:MTLResourceStorageModeShared];
    
    b_fc_in = [device newBufferWithLength:sizeof(float) * 512 * BATCH_SIZE 
                                      options:MTLResourceStorageModeShared];
    
    b_fc_out = [device newBufferWithLength:sizeof(float) * 512 * BATCH_SIZE 
                                       options:MTLResourceStorageModeShared];
}

void initialize_network_offsets(int* offsets) {
    // Convolution layers (0-12)
    offsets[0] = 3 * 3 * INPUT_in_dim[0] * OUTPUT_in_dim[0] + OUTPUT_in_dim[0];
    offsets[1] = 3 * 3 * INPUT_in_dim[1] * OUTPUT_in_dim[1] + OUTPUT_in_dim[1];
    offsets[2] = 3 * 3 * INPUT_in_dim[3] * OUTPUT_in_dim[3] + OUTPUT_in_dim[3];
    offsets[3] = 3 * 3 * INPUT_in_dim[4] * OUTPUT_in_dim[4] + OUTPUT_in_dim[4];
    offsets[4] = 3 * 3 * INPUT_in_dim[6] * OUTPUT_in_dim[6] + OUTPUT_in_dim[6];
    offsets[5] = 3 * 3 * INPUT_in_dim[7] * OUTPUT_in_dim[7] + OUTPUT_in_dim[7];
    offsets[6] = 3 * 3 * INPUT_in_dim[8] * OUTPUT_in_dim[8] + OUTPUT_in_dim[8];
    offsets[7] = 3 * 3 * INPUT_in_dim[10] * OUTPUT_in_dim[10] + OUTPUT_in_dim[10];
    offsets[8] = 3 * 3 * INPUT_in_dim[11] * OUTPUT_in_dim[11] + OUTPUT_in_dim[11];
    offsets[9] = 3 * 3 * INPUT_in_dim[12] * OUTPUT_in_dim[12] + OUTPUT_in_dim[12];
    offsets[10] = 3 * 3 * INPUT_in_dim[14] * OUTPUT_in_dim[14] + OUTPUT_in_dim[14];
    offsets[11] = 3 * 3 * INPUT_in_dim[15] * OUTPUT_in_dim[15] + OUTPUT_in_dim[15];
    offsets[12] = 3 * 3 * INPUT_in_dim[16] * OUTPUT_in_dim[16] + OUTPUT_in_dim[16];

    // FC layers (13-14)
    offsets[13] = INPUT_in_dim[18] * OUTPUT_in_dim[18] + OUTPUT_in_dim[18];
    offsets[14] = INPUT_in_dim[19] * OUTPUT_in_dim[19] + OUTPUT_in_dim[19];
}

void process_single_batch(const int* network_offsets, float* result, int batch_start_idx, int current_batch_size) {
    int image_offset = 32 * 32 * 3 * batch_start_idx;
    int network_offset = 0;

    // Convolution layer 1
    convolution_metal(b_imgs, b_conv_out, INPUT_in_dim[0], OUTPUT_in_dim[0], NBYN[0], 
                     image_offset, network_offset);
    image_offset = 0;
    network_offset += network_offsets[0];

    // Convolution layer 2
    convolution_metal(b_conv_out, b_conv_in, INPUT_in_dim[1], OUTPUT_in_dim[1], 
                     NBYN[1], image_offset, network_offset);
    network_offset += network_offsets[1];
    max_pooling_metal(b_conv_in, b_pool_out, INPUT_in_dim[2], NBYN[2] * 2);

    // Convolution block 1
    convolution_metal(b_pool_out, b_conv_out, INPUT_in_dim[3], OUTPUT_in_dim[3], 
                     NBYN[3], image_offset, network_offset);
    network_offset += network_offsets[2];
    convolution_metal(b_conv_out, b_conv_in, INPUT_in_dim[4], OUTPUT_in_dim[4], 
                     NBYN[4], image_offset, network_offset);
    network_offset += network_offsets[3];
    max_pooling_metal(b_conv_in, b_pool_out, INPUT_in_dim[5], NBYN[5] * 2);

    // Convolution block 2
    convolution_metal(b_pool_out, b_conv_out, INPUT_in_dim[6], OUTPUT_in_dim[6], 
                     NBYN[6], image_offset, network_offset);
    network_offset += network_offsets[4];
    convolution_metal(b_conv_out, b_conv_in, INPUT_in_dim[7], OUTPUT_in_dim[7], 
                     NBYN[7], image_offset, network_offset);
    network_offset += network_offsets[5];
    convolution_metal(b_conv_in, b_conv_out, INPUT_in_dim[8], OUTPUT_in_dim[8], 
                     NBYN[8], image_offset, network_offset);
    network_offset += network_offsets[6];
    max_pooling_metal(b_conv_out, b_pool_out, INPUT_in_dim[9], NBYN[9] * 2);

    // Convolution block 3
    convolution_metal(b_pool_out, b_conv_out, INPUT_in_dim[10], OUTPUT_in_dim[10], 
                     NBYN[10], image_offset, network_offset);
    network_offset += network_offsets[7];
    convolution_metal(b_conv_out, b_conv_in, INPUT_in_dim[11], OUTPUT_in_dim[11], 
                     NBYN[11], image_offset, network_offset);
    network_offset += network_offsets[8];
    convolution_metal(b_conv_in, b_conv_out, INPUT_in_dim[12], OUTPUT_in_dim[12], 
                     NBYN[12], image_offset, network_offset);
    network_offset += network_offsets[9];
    max_pooling_metal(b_conv_out, b_pool_out, INPUT_in_dim[13], NBYN[13] * 2);

    // Convolution block 4
    convolution_metal(b_pool_out, b_conv_out, INPUT_in_dim[14], OUTPUT_in_dim[14], 
                     NBYN[14], image_offset, network_offset);
    network_offset += network_offsets[10];
    convolution_metal(b_conv_out, b_conv_in, INPUT_in_dim[15], OUTPUT_in_dim[15], 
                     NBYN[15], image_offset, network_offset);
    network_offset += network_offsets[11];
    convolution_metal(b_conv_in, b_conv_out, INPUT_in_dim[16], OUTPUT_in_dim[16], 
                     NBYN[16], image_offset, network_offset);
    network_offset += network_offsets[12];
    max_pooling_metal(b_conv_out, b_pool_out, INPUT_in_dim[17], NBYN[17] * 2);

    // FC layers
    fc_layer_metal(b_pool_out, b_fc_out, INPUT_in_dim[18], OUTPUT_in_dim[18], network_offset);
    network_offset += network_offsets[13];
    fc_layer_metal(b_fc_out, b_fc_in, INPUT_in_dim[19], OUTPUT_in_dim[19], network_offset);
    network_offset += network_offsets[14];
    fc_layer_metal(b_fc_in, b_fc_out, INPUT_in_dim[20], OUTPUT_in_dim[20], network_offset);

    // 결과 복사
    float* fc_outputs_ptr = (float*)[b_fc_out contents];
    memcpy(result, fc_outputs_ptr, sizeof(float) * 10 * current_batch_size);
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
    float* result = (float*)malloc(sizeof(float) * 10 * BATCH_SIZE);

    // 배치 계산
    int full_batches = num_images / BATCH_SIZE;
    int remaining_images = num_images % BATCH_SIZE;

    // 시간 측정 시작
    clock_t start = clock();

    // 전체 배치 처리
    for (int i = 0; i < full_batches; i++) {
        process_single_batch(network_offsets, result, i * BATCH_SIZE, BATCH_SIZE);
        process_results(result, labels, confidences, i * BATCH_SIZE, BATCH_SIZE);
    }

    // 남은 이미지 처리
    if (remaining_images > 0) {
        process_single_batch(network_offsets, result, full_batches * BATCH_SIZE, remaining_images);
        process_results(result, labels, confidences, full_batches * BATCH_SIZE, remaining_images);
    }

    // 시간 측정 종료
    gettimeofday(&end_time, NULL);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + 
                         (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    printf("Elapsed time: %.6f seconds\n", elapsed_time);

    // 메모리 해제
    free(result);
    [b_imgs release];
    [b_net release];
    [b_conv_in release];
    [b_conv_out release];
    [b_pool_out release];
    [b_fc_in release];
    [b_fc_out release];
    [p_conv release];
    [p_pool release];
    [p_fc release];
    [f_conv release];
    [f_pool release];
    [f_fc release];
    [library release];
    [queue release];
    [device release];
}
