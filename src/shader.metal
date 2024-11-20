// shader.metal
#include <metal_stdlib>
using namespace metal;

kernel void convolution_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* filter [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    device const int* params [[buffer(4)]],
    uint3 id [[thread_position_in_grid]]
) {
    // 각 스레드는 하나의 출력 픽셀을 담당
    const int outNeuron = id.z; // 출력 채널 (0 ~ outDim-1)
    const int row = id.y; // y 위치 (0 ~ nbyn-1)
    const int col = id.x; // x 위치 (0 ~ nbyn-1)
    
    // params에서 설정값 읽기
    const int inDim = params[0]; // 입력 채널 수
    const int outDim = params[1]; // 출력 채널 수
    const int nbyn = params[2]; // 이미지 크기
    const int offset = nbyn * nbyn;
    
    float sum = 0.0f;
    
    // 각 입력 채널에 대해
    for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
        // 3x3 필터 연산
        for (int fRow = 0; fRow < 3; ++fRow) {
            for (int fCol = 0; fCol < 3; ++fCol) {
                // 입력 위치 계산
                int x = col + fCol - 1; // -1은 패딩
                int y = row + fRow - 1;
                
                // 이미지 경계 체크
                if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
                    // 입력 데이터 인덱스:
                    // (입력채널 * 이미지크기^2) + (y * 이미지폭) + x
                    const int inputIdx = inNeuron * offset + y * nbyn + x;

                    // 필터 가중치 인덱스:
                    // (출력채널 * 입력채널 * 9) + (필터행 * 3) + 필터열
                    const int filterIdx = (outNeuron * inDim + inNeuron) * 9 + fRow * 3 + fCol;

                    // 컨볼루션 연산 누적
                    sum += input[inputIdx] * filter[filterIdx];
                }
            }
        }
    }
    
    // 출력 인덱스:
    // (출력채널 * 이미지크기^2) + (행 * 이미지폭) + 열
    const int outIdx = outNeuron * offset + row * nbyn + col;

    // 바이어스 추가 및 ReLU 활성화 함수 적용
    sum += bias[outNeuron];
    output[outIdx] = sum > 0.0f ? sum : 0.0f; // ReLU
}
