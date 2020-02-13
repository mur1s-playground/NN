#include "ReLu.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"

__global__ void reluKernel(
	const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int filters) {

	float* input = (float*)&device_data_input[input_position];
	float* output = (float*)&device_data_output[output_position];

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < filters * input_w * input_h) {
		if (input[i] < 0) {
			output[i] = 0;
		} else {
			output[i] = input[i];
		}
	}
}

void launch_relu(
	const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int filters) {

	unsigned int size = input_w * input_h * filters;

	cudaError_t err = cudaSuccess;

	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

	reluKernel << <blocksPerGrid, threadsPerBlock >> > (
		device_data_input, input_position, input_w, input_h,
		device_data_output, output_position,
		filters);

	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed in reluKernel (error code %s)\n", cudaGetErrorString(err));
	}
}