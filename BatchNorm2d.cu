#include "BatchNorm2d.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"

__global__ void batchnorm2dKernel(const unsigned int* device_data_parameters, const unsigned int parameters_position_weights, const unsigned int parameters_position_bias,
	const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int filters) {

	float* weights = (float*)&device_data_parameters[parameters_position_weights];
	float* bias = (float*)&device_data_parameters[parameters_position_bias];

	float* input = (float*)&device_data_input[input_position];
	float* output = (float*)&device_data_output[output_position];

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < filters * input_w * input_h) {
		unsigned int current_filter = i / (input_w * input_h);
		output[i] = (input[i] * weights[current_filter]) + bias[current_filter];
	}
}

void launch_batchnorm2d(const unsigned int* device_data_parameters, const unsigned int parameters_position_weights, const unsigned int parameters_position_bias,
		const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h,
		unsigned int* device_data_output, const unsigned int output_position, const unsigned int filters) {

		unsigned int size = input_w * input_h * filters;

		cudaError_t err = cudaSuccess;

		int threadsPerBlock = 256;
		int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

		batchnorm2dKernel<<<blocksPerGrid, threadsPerBlock>>> (device_data_parameters, parameters_position_weights, parameters_position_bias,
			device_data_input, input_position, input_w, input_h,
			device_data_output, output_position,
			filters);

		err = cudaGetLastError();

		if (err != cudaSuccess) {
			fprintf(stderr, "Failed in batchnorm2dKernel (error code %s)\n", cudaGetErrorString(err));
		}
}

void batchnorm2d_init(struct bn2d* bn, const unsigned int filters, const unsigned int position_weights, const unsigned int position_bias) {
	bn->filters = filters;
	bn->position_weights = position_weights;
	bn->position_bias = position_bias;
}