// main.mm
#include <stdio.h>
#include <stdlib.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

int main() {
    @autoreleasepool {
        // Metal 디바이스 초기화
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal is not supported on this device\n");
            return -1;
        }
        printf("Using device: %s\n", [device.name UTF8String]);
        
        // 테스트 데이터 준비
        const int SIZE = 32;
        float input[SIZE * SIZE];
        float filter[3 * 3];
        float bias = 1.0f;
        float output[SIZE * SIZE];
        
        // 테스트 데이터 초기화
        for (int i = 0; i < SIZE * SIZE; i++) {
            input[i] = 1.0f;  // 모든 픽셀을 1로 설정
        }
        for (int i = 0; i < 9; i++) {
            filter[i] = 1.0f;  // 모든 필터 값을 1로 설정
        }
        
        // Metal 버퍼 생성
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input
                                                       length:SIZE * SIZE * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithBytes:output
                                                        length:SIZE * SIZE * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> filterBuffer = [device newBufferWithBytes:filter
                                                        length:9 * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> biasBuffer = [device newBufferWithBytes:&bias
                                                      length:sizeof(float)
                                                     options:MTLResourceStorageModeShared];
        
        // Metal 라이브러리 로드
        NSError* error = nil;
        NSString* libraryPath = [[NSBundle mainBundle] pathForResource:@"shader" ofType:@"metallib"];
        if (!libraryPath) {
            libraryPath = @"build/shader.metallib";  // 현재 디렉토리에서 찾기
        }
        
        id<MTLLibrary> library = [device newLibraryWithFile:libraryPath error:&error];
        if (!library) {
            fprintf(stderr, "Failed to load Metal library: %s\n", 
                    error ? [[error localizedDescription] UTF8String] : "unknown error");
            return -1;
        }
        
        // 커널 함수 설정
        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"basic_convolution"];
        if (!kernelFunction) {
            fprintf(stderr, "Failed to find kernel function\n");
            return -1;
        }
        
        // Pipeline state 생성
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!pipelineState) {
            fprintf(stderr, "Failed to create pipeline state: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown error");
            return -1;
        }
        
        // Command queue 생성
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            fprintf(stderr, "Failed to create command queue\n");
            return -1;
        }
        
        // Command buffer와 encoder 생성
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        if (!commandBuffer) {
            fprintf(stderr, "Failed to create command buffer\n");
            return -1;
        }
        
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            fprintf(stderr, "Failed to create compute encoder\n");
            return -1;
        }
        
        // Compute command 설정
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:filterBuffer offset:0 atIndex:2];
        [encoder setBuffer:biasBuffer offset:0 atIndex:3];
        
        // Thread 구성 설정
        MTLSize gridSize = MTLSizeMake(SIZE, SIZE, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        
        // Compute command 실행
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        
        // 실행 및 완료 대기
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // 결과 확인
        float* resultData = (float*)outputBuffer.contents;
        printf("First few results: %f %f %f\n", resultData[0], resultData[1], resultData[2]);
    }
    
    return 0;
}
