#ifndef CNN_METAL_H
#define CNN_METAL_H

#ifdef __cplusplus
extern "C" {
#endif

// CNN 초기화 함수
void cnn_init(float* images, float* network, int* labels, float* confidences,
			  int num_images);

// CNN 실행 함수
void cnn(float* images, float* network, int* labels, float* confidences,
		 int num_images);

// Convolution 연산 함수
void convolution_metal(void* inputs, void* outputs, int inDim, int outDim,
					   int nbyn, int image_offset, int network_offset);

// Max pooling 연산 함수
void max_pooling_metal(void* input, void* output, int DIM, int nbyn);

// Fully connected layer 연산 함수
void fc_layer_metal(void* input, void* output, int inDim, int outDim,
					int network_offset);

// 헬퍼 함수들
void softmax(float* input, int N);
int find_max(float* input, int classNum);
void process_results(float* result, int* labels, float* confidences,
					 int batch_start_idx, int current_batch_size);

#ifdef __cplusplus
}
#endif

#endif	// CNN_METAL_H
