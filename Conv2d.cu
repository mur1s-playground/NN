#include "Conv2d.hpp"

#include "cuda_runtime.h"

__global__ void conv2dKernel(	const unsigned int *device_data_parameters, const unsigned int parameters_position, 
								const unsigned int *device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h,
								unsigned int *device_data_output, const unsigned int output_position, const unsigned int output_w, const unsigned int output_h,
								const unsigned int channels, const unsigned int filters, const unsigned int kernel_size_x, const unsigned int kernel_size_y, 
								const unsigned int stride_x, const unsigned int stride_y, const unsigned int padding_x, const unsigned int padding_y, 
								const unsigned int dilation_x, const unsigned int dilation_y
	) {
		float* parameters = (float*) &device_data_parameters[parameters_position];
		float* input = (float *) &device_data_input[input_position];
		float* output = (float *) &device_data_output[output_position];

		int i = blockIdx.x * blockDim.x + threadIdx.x;

		if (i < filters * output_w * output_h) {
			int current_filter_id = i / (output_w * output_h);
			int current_position = i % (output_w * output_h);
			int output_pos_x = current_position % output_w;
			int output_pos_y = current_position / output_w;

			int input_pad_x = (kernel_size_x / 2) - padding_x;
			int input_pad_y = (kernel_size_y / 2) - padding_y;

			int input_pos_x = ((output_pos_x+input_pad_x)*stride_x) - (kernel_size_x / 2);
			int input_pos_y = ((output_pos_y+input_pad_y)*stride_y) - (kernel_size_y / 2);

			float output_f = 0;
			for (int c = 0; c < channels; c++) {
				for (int k_x = 0; k_x < kernel_size_x; k_x += dilation_x) {
					for (int k_y = 0; k_y < kernel_size_y; k_y += dilation_y) {
						float param = parameters[current_filter_id * (kernel_size_x * kernel_size_y * channels) + c * (kernel_size_x * kernel_size_y) + k_y * kernel_size_x + k_x];
						float input_f = 0;
						if (input_pos_x + k_x >= 0 && input_pos_y + k_y >= 0 && input_pos_x + k_x < input_w && input_pos_y + k_y < input_h) {
							input_f = input[(input_pos_y + k_y) * (input_w * channels) + (input_pos_x + k_x) * channels + c];
						}
						output_f += (param * input_f);
					}
				}
			}
			output[current_filter_id * (output_w * output_h) + output_pos_y * output_w + output_pos_x] = output_f;
		}
}

void launch_conv2d(const unsigned int* device_data_parameters, const unsigned int parameters_position,
	const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h,
	unsigned int* device_data_output, const unsigned int output_position, 
	struct conv2d *c) {

	struct vector3<unsigned int> output_dim = conv2d_get_output_dim(c, struct vector2<unsigned int> (input_w, input_h));
	unsigned int output_size = output_dim[0] * output_dim[1] * output_dim[2];

	cudaError_t err = cudaSuccess;

	int threadsPerBlock = 256;
	int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

	conv2dKernel<<<blocksPerGrid, threadsPerBlock>>>(device_data_parameters, parameters_position, device_data_input, input_position, input_w, input_h, device_data_output, output_position, output_dim[0], output_dim[1],
													c->channels, c->filters, c->kernel_size[0], c->kernel_size[1], c->stride[0], c->stride[1], 
													c->padding[0], c->padding[1], c->dilation[0], c->dilation[1]);
	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed in conv2dKernel (error code %s)\n", cudaGetErrorString(err));
	}
}

void conv2d_init(struct conv2d* c, unsigned int channels, unsigned int filters, struct vector2<unsigned int> kernel_size, struct vector2<unsigned int> stride, struct vector2<unsigned int> padding, struct vector2<unsigned int> dilation, unsigned int parameters_position) {
	c->channels = channels;
	c->filters = filters;
	c->kernel_size = kernel_size;
	c->stride = stride;
	c->padding = padding;
	c->dilation = dilation;
	c->parameters_position = parameters_position;
}

struct vector3<unsigned int> conv2d_get_output_dim(struct conv2d* c, struct vector2<unsigned int> input_dim) {
	struct vector3<unsigned int> output((c->padding[0] - (c->kernel_size[0] / 2) + input_dim[0]) / c->stride[0], (c->padding[1] - (c->kernel_size[1] / 2) + input_dim[1]) / c->stride[1], c->filters);
	return output;
}