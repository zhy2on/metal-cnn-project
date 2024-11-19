// shader.metal
#include <metal_stdlib>
using namespace metal;

kernel void basic_convolution(
    const device float* input [[buffer(0)]],     // 32x32 입력 이미지
    device float* output [[buffer(1)]],          // 32x32 출력 이미지
    const device float* filter [[buffer(2)]],    // 3x3 필터
    const device float* bias [[buffer(3)]],      // 단일 bias 값
    uint2 pos [[thread_position_in_grid]])       // 현재 스레드 위치
{
    const int size = 32;  // 이미지 크기
    const int x = pos.x;
    const int y = pos.y;
    
    // 경계 체크
    if (x >= size || y >= size) return;
    
    float sum = 0.0f;
    
    // 3x3 필터 적용
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int ix = x + kx - 1;
            int iy = y + ky - 1;
            
            if (ix >= 0 && ix < size && iy >= 0 && iy < size) {
                sum += input[iy * size + ix] * filter[ky * 3 + kx];
            }
        }
    }
    
    // bias 더하고 ReLU 적용
    sum += *bias;
    sum = max(0.0f, sum);
    
    // 결과 저장
    output[y * size + x] = sum;
}
