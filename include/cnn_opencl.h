#ifndef CNN_OPENCL_H
#define CNN_OPENCL_H

// Metal 구현에 필요한 유틸리티 함수들 선언
void cnn_init();
void cnn(float* images, float** network, int* labels, float* confidences,
		 int num_images);

// 각 레이어 연산 함수 선언
static void max_pooling(float* input, float* output, int DIM, int nbyn);
void fc_layer(float* input, float* output, float* weights, float* biases,
			  int inDim, int outDim);
static void softmax(float* input, int N);
static int find_max(float* input, int classNum);

#endif	// CNN_OPENCL_H
