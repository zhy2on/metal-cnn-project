#ifndef CNN_OPENCL_H
#define CNN_OPENCL_H

#ifdef __cplusplus
extern "C" {
#endif

void cnn_init();
void cnn(float* images, float* network, int* labels, float* confidences,
		 int num_of_image);

#ifdef __cplusplus
}
#endif

#endif
