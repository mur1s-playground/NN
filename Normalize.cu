#include "Normalize.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "resize.hpp"
#include "stdio.h"

__global__ void normalizeKernel(const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h, const unsigned int channels,
	unsigned int* device_data_output, const unsigned int output_position, struct vector3<float> mean, struct vector3<float> std) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < input_w * input_h * channels) {
		int current_channel = i / (input_w * input_h);
		int current_idx = i % (input_w * input_h);
		int current_x = (current_idx % input_w);
		int current_y = (current_idx / input_w);

		unsigned char* input = (unsigned char*)&device_data_input[input_position];
		float* output = (float*) &device_data_output[output_position];

		output[current_y * (input_w * channels) + current_x * channels + current_channel] = 
				((input[current_y * (input_w * channels) + current_x * channels + current_channel] / 255.0f) - mean[current_channel]) / std[current_channel];
	}
}

void launch_normalize(const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h, const unsigned int channels,
	unsigned int* device_data_output, const unsigned int output_position, const struct vector3<float> mean, const struct vector3<float> std) {

	unsigned int size = input_w * input_h * channels;

	cudaError_t err = cudaSuccess;

	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

	normalizeKernel<<<blocksPerGrid, threadsPerBlock>>> (device_data_input, input_position, input_w, input_h, channels, device_data_output, output_position, mean, std);
	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed in normalizeKernel (error code %s)\n", cudaGetErrorString(err));
	}
}