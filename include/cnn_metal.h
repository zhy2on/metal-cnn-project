// cnn_metal.h
#ifndef CNN_METAL_H
#define CNN_METAL_H

#ifdef __cplusplus
extern "C" {
#endif

void cnn_init();
void cnn(float* images, float* network, int* labels, float* confidences,
		 int num_of_image);

// 상수 정의
#define BATCH_SIZE 64
#define INPUT_SIZE 32
#define NUM_CHANNELS 3
#define NUM_CLASSES 10

// CNN 구조 상수
static const int INPUT_DIM[] = {
	3,	 64,  64,		 // Conv block 1 (input -> 64 -> 64)
	64,	 128, 128,		 // Conv block 2 (64 -> 128 -> 128)
	128, 256, 256, 256,	 // Conv block 3 (128 -> 256 -> 256 -> 256)
	256, 512, 512, 512,	 // Conv block 4 (256 -> 512 -> 512 -> 512)
	512, 512, 512, 512,	 // Conv block 5 (512 -> 512 -> 512 -> 512)
	512, 512, 512		 // FC layers (512 -> 512 -> 512)
};

static const int OUTPUT_DIM[] = {
	64,	 64,  64,		 // Conv block 1 output channels
	128, 128, 128,		 // Conv block 2 output channels
	256, 256, 256, 256,	 // Conv block 3 output channels
	512, 512, 512, 512,	 // Conv block 4 output channels
	512, 512, 512, 512,	 // Conv block 5 output channels
	512, 512, 10		 // FC layers (final 10 for classification)
};

static const int NBYN[] = {
	32, 32, 16,		// Conv block 1: 32x32 -> pool -> 16x16
	16, 16, 8,		// Conv block 2: 16x16 -> pool -> 8x8
	8,	8,	8,	4,	// Conv block 3: 8x8 -> pool -> 4x4
	4,	4,	4,	2,	// Conv block 4: 4x4 -> pool -> 2x2
	2,	2,	2,	1,	// Conv block 5: 2x2 -> pool -> 1x1
	1,	1,	1		// FC layers: all 1x1 convolutions
};

static const int NETWORK_OFFSETS[] = {
	3 * 3 * 3 * 64 + 64,	  // Conv1_1
	3 * 3 * 64 * 64 + 64,	  // Conv1_2
	3 * 3 * 64 * 128 + 128,	  // Conv2_1
	3 * 3 * 128 * 128 + 128,  // Conv2_2
	3 * 3 * 128 * 256 + 256,  // Conv3_1
	3 * 3 * 256 * 256 + 256,  // Conv3_2
	3 * 3 * 256 * 256 + 256,  // Conv3_3
	3 * 3 * 256 * 512 + 512,  // Conv4_1
	3 * 3 * 512 * 512 + 512,  // Conv4_2
	3 * 3 * 512 * 512 + 512,  // Conv4_3
	3 * 3 * 512 * 512 + 512,  // Conv5_1
	3 * 3 * 512 * 512 + 512,  // Conv5_2
	3 * 3 * 512 * 512 + 512,  // Conv5_3
	512 * 512 + 512,		  // FC6
	512 * 512 + 512			  // FC7
};

#ifdef __cplusplus
}
#endif

#endif
