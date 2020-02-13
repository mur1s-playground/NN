#include "MaxPool2d.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"

__global__ void maxpool2dKernel(
	const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_w, const unsigned int output_h, 
	const unsigned int filters, struct vector2<unsigned int>kernel_size, struct vector2<unsigned int>stride, struct vector2<unsigned int>padding, struct vector2<unsigned int>dilation) {

	float* input = (float*)&device_data_input[input_position];
	float* output = (float*)&device_data_output[output_position];

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < filters * output_w * output_h) {
		int current_filter_id = i / (output_w * output_h);
		int current_position = i % (output_w * output_h);
		int output_pos_x = current_position % output_w;
		int output_pos_y = current_position / output_w;

		int input_pad_x = (kernel_size[0] / 2) - padding[0];
		int input_pad_y = (kernel_size[1] / 2) - padding[1];

		int input_pos_x = ((output_pos_x + input_pad_x) * stride[0]) - (kernel_size[0] / 2);
		int input_pos_y = ((output_pos_y + input_pad_y) * stride[1]) - (kernel_size[1] / 2);

		float output_f = 0;
		for (int k_x = 0; k_x < kernel_size[0]; k_x += dilation[0]) {
			for (int k_y = 0; k_y < kernel_size[1]; k_y += dilation[1]) {
				float input_f = 0;
				if (input_pos_x + k_x >= 0 && input_pos_y + k_y >= 0 && input_pos_x + k_x < input_w && input_pos_y + k_y < input_h) {
					input_f = input[current_filter_id * (input_w * input_h) + (input_pos_y + k_y) * (input_w)+(input_pos_x + k_x)];
					if (input_f > output_f) output_f = input_f;
				}
			}
		}
		output[current_filter_id * (output_w * output_h) + output_pos_y * output_w + output_pos_x] = output_f;
	}
}

void launch_maxpool2d(
	const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h,
	unsigned int* device_data_output, const unsigned int output_position,
	struct maxpool2d* m) {

	struct vector3<unsigned int> output_dim = maxpool2d_get_output_dim(m, struct vector2<unsigned int>(input_w, input_h));
	unsigned int output_size = output_dim[0] * output_dim[1] * output_dim[2];

	cudaError_t err = cudaSuccess;

	int threadsPerBlock = 256;
	int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

	maxpool2dKernel <<<blocksPerGrid, threadsPerBlock>>>(device_data_input, input_position, input_w, input_h, device_data_output, output_position, output_dim[0], output_dim[1],
		m->filters, m->kernel_size, m->stride,
		m->padding, m->dilation);
	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed in maxpool2dKernel (error code %s)\n", cudaGetErrorString(err));
	}
}

void maxpool2d_init(struct maxpool2d* m, unsigned int filters, struct vector2<unsigned int> kernel_size, struct vector2<unsigned int> stride, struct vector2<unsigned int> padding, struct vector2<unsigned int> dilation) {
	m->filters = filters;
	m->kernel_size = kernel_size;
	m->stride = stride;
	m->padding = padding;
	m->dilation = dilation;
}

struct vector3<unsigned int> maxpool2d_get_output_dim(struct maxpool2d* m, struct vector2<unsigned int> input_dim) {
	struct vector3<unsigned int> output((m->padding[0] - (m->kernel_size[0] / 2) + input_dim[0]) / m->stride[0], (m->padding[1] - (m->kernel_size[1] / 2) + input_dim[1]) / m->stride[1], m->filters);
	return output;
}