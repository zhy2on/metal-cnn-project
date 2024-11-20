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
        
        // 3. Metal 라이브러리(셰이더 코드) 로드
        NSError* error = nil;
        NSURL *libraryURL = [NSURL fileURLWithPath:@"build/shader.metallib"];
		library = [device newLibraryWithURL:libraryURL error:&error];
        if (!library) {
            fprintf(stderr, "Failed to load Metal library: %s\n", 
                    error ? [[error localizedDescription] UTF8String] : "unknown error");
            return;
        }
        
        // 4. 컴퓨트 파이프라인 상태 생성
        id<MTLFunction> convFunction = [library newFunctionWithName:@"convolution_kernel"];
        convPipelineState = [device newComputePipelineStateWithFunction:convFunction error:&error];
        if (!convPipelineState) {
            fprintf(stderr, "Failed to create pipeline state\n");
            return;
        }
    }
}

// Metal 구현 convolution
static void convolution_metal(float* inputs, float* outputs, float* filter, float* biases, 
                            int inDim, int outDim, int nbyn) {
    @autoreleasepool {
        // 1. 입출력 버퍼 크기 계산
        NSUInteger inputSize = nbyn * nbyn * inDim * sizeof(float);
        NSUInteger outputSize = nbyn * nbyn * outDim * sizeof(float);
        NSUInteger filterSize = 3 * 3 * inDim * outDim * sizeof(float);
        NSUInteger biasSize = outDim * sizeof(float);
        
        // 2. Metal 버퍼 생성
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:inputs length:inputSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithBytes:outputs length:outputSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> filterBuffer = [device newBufferWithBytes:filter length:filterSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> biasBuffer = [device newBufferWithBytes:biases length:biasSize options:MTLResourceStorageModeShared];
        
        // 3. 파라미터 버퍼 생성
        int params[] = {inDim, outDim, nbyn};
        id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:params length:sizeof(params) options:MTLResourceStorageModeShared];
        
        // 4. 커맨드 버퍼와 인코더 생성
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // 5. 커널 설정
        [encoder setComputePipelineState:convPipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:filterBuffer offset:0 atIndex:2];
        [encoder setBuffer:biasBuffer offset:0 atIndex:3];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:4];
        
        // 6. 스레드 그룹 크기 설정
        MTLSize threadGroupSize = MTLSizeMake(8, 8, 1);
        MTLSize gridSize = MTLSizeMake(nbyn, nbyn, outDim);
        
        // 7. GPU에서 실행
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // 8. 결과 복사
        memcpy(outputs, outputBuffer.contents, outputSize);
    }
}

static void max_pooling_metal(float* input, float* output, int DIM, int nbyn) {
    @autoreleasepool {
        // 입력과 출력 버퍼 크기 계산
        NSUInteger inSize = nbyn * nbyn * DIM * sizeof(float);          // 입력 크기
        NSUInteger outSize = (nbyn/2) * (nbyn/2) * DIM * sizeof(float); // 출력 크기
        
        // 파라미터 설정
        int params[] = {DIM, nbyn/2}; // {채널 수, 출력 크기}
        NSUInteger paramsSize = sizeof(params);
        
        // Metal 버퍼 생성
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input 
                                                      length:inSize 
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithBytes:output 
                                                       length:outSize 
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:params 
                                                       length:paramsSize 
                                                      options:MTLResourceStorageModeShared];
        
        // 커맨드 버퍼와 인코더 생성
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // max pooling 커널용 파이프라인 상태 설정
        id<MTLFunction> poolFunction = [library newFunctionWithName:@"max_pooling_kernel"];
        id<MTLComputePipelineState> poolPipelineState = [device newComputePipelineStateWithFunction:poolFunction error:nil];
        [encoder setComputePipelineState:poolPipelineState];
        
        // 버퍼 설정
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
        
        // 스레드 그룹 크기와 그리드 크기 설정
        MTLSize threadGroupSize = MTLSizeMake(8, 8, 1);
        MTLSize gridSize = MTLSizeMake(nbyn/2, nbyn/2, DIM);
        
        // 커널 실행
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        
        // 실행 및 완료 대기
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // 결과 복사
        memcpy(output, outputBuffer.contents, outSize);
    }
}

// cnn_seq.cpp에서 가져온 나머지 함수들
static void max_pooling(float* input, float* output, int DIM, int nbyn) {
    float max, temp;
    for (int n = 0; n < DIM; ++n) {
        for (int row = 0; row < nbyn; row += 2) {
            for (int col = 0; col < nbyn; col += 2) {
                max = 0;
                for (int y = 0; y < 2; ++y) {
                    for (int x = 0; x < 2; ++x) {
                        temp = input[nbyn * (row + y) + col + x];
                        if (max < temp) max = temp;
                    }
                }
                *(output++) = max;
            }
        }
        input += nbyn * nbyn;
    }
}

static void fc_layer(float* input, float* output, float* weights, float* biases, int inDim, int outDim) {
    for (int outNeuron = 0; outNeuron < outDim; ++outNeuron) {
        float sum = 0;
        for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
            sum += input[inNeuron] * weights[outNeuron * inDim + inNeuron];
        }
        sum += biases[outNeuron];
        output[outNeuron] = sum > 0 ? sum : 0;  // ReLU
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
    
    // weights와 biases 링크
    float* w[21];
    float* b[21];
    int offset = 0;
    
    for (int i = 0; i < 17; ++i) {
        if (i == 2 || i == 5 || i == 9 || i == 13) {
            i++;
        }
        w[i] = network + offset;
        offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];
        b[i] = network + offset;
        offset += OUTPUT_DIM[i];
    }
    for (int i = 18; i < 21; ++i) {
        w[i] = network + offset;
        offset += INPUT_DIM[i] * OUTPUT_DIM[i];
        b[i] = network + offset;
        offset += OUTPUT_DIM[i];
    }
    
    // layer 메모리 할당
    float* layer[21];
    for (int i = 0; i < 21; ++i) {
        layer[i] = (float*)malloc(sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i]);
        if (layer[i] == NULL) {
            perror("malloc error");
            return;
        }
    }
    
    // CNN 실행
    for (int i = 0; i < num_of_image; ++i) {
        // Metal로 구현된 convolution 사용
        convolution_metal(images, layer[0], w[0], b[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
        convolution_metal(layer[0], layer[1], w[1], b[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
        max_pooling_metal(layer[1], layer[2], INPUT_DIM[2], NBYN[2] * 2);
        
        // 나머지 레이어들도 같은 패턴으로 처리
        convolution_metal(layer[2], layer[3], w[3], b[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
        convolution_metal(layer[3], layer[4], w[4], b[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
        max_pooling_metal(layer[4], layer[5], INPUT_DIM[5], NBYN[5] * 2);
        
        convolution_metal(layer[5], layer[6], w[6], b[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
        convolution_metal(layer[6], layer[7], w[7], b[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
        convolution_metal(layer[7], layer[8], w[8], b[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
        max_pooling_metal(layer[8], layer[9], INPUT_DIM[9], NBYN[9] * 2);
        
        convolution_metal(layer[9], layer[10], w[10], b[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
        convolution_metal(layer[10], layer[11], w[11], b[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
        convolution_metal(layer[11], layer[12], w[12], b[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
        max_pooling_metal(layer[12], layer[13], INPUT_DIM[13], NBYN[13] * 2);
        
        convolution_metal(layer[13], layer[14], w[14], b[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
        convolution_metal(layer[14], layer[15], w[15], b[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
        convolution_metal(layer[15], layer[16], w[16], b[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
        max_pooling_metal(layer[16], layer[17], INPUT_DIM[17], NBYN[17] * 2);
        
        fc_layer(layer[17], layer[18], w[18], b[18], INPUT_DIM[18], OUTPUT_DIM[18]);
        fc_layer(layer[18], layer[19], w[19], b[19], INPUT_DIM[19], OUTPUT_DIM[19]);
        fc_layer(layer[19], layer[20], w[20], b[20], INPUT_DIM[20], OUTPUT_DIM[20]);
        
        softmax(layer[20], 10);
        
        labels[i] = find_max(layer[20], 10);
        confidences[i] = layer[20][labels[i]];
        images += 32 * 32 * 3;
    }
    
    // 메모리 해제
    for (int i = 0; i < 21; ++i) {
        free(layer[i]);
    }
    
    end = clock();
    printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLOCKS_PER_SEC);
}
