#include "Bottleneck.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"

__global__ void addKernel(const unsigned int* device_data_input,
    const unsigned int input_position, const unsigned int width, const unsigned int height, const unsigned int channels,
    unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
    const unsigned int type
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width * height * channels) {
        int current_channel = i / (width * height);
        int current_idx = i % (width * height);
        int current_x = (current_idx % width);
        int current_y = (current_idx / width);

        if (type == 0) {
            unsigned char* input = (unsigned char*)&device_data_input[input_position];
            unsigned char* output = (unsigned char*)&device_data_output[output_position];

            if (current_channel >= channels) {
            } else {
                output[(current_y) * (output_width * output_channels) + (current_x) * output_channels + current_channel] += input[current_y * (width * channels) + current_x * channels + current_channel];
            }
        } else {
            float* input = (float*)&device_data_input[input_position];
            float* output = (float*)&device_data_output[output_position];

            if (current_channel >= channels) {
            } else {
                output[(current_y) * (output_width * output_channels) + (current_x)*output_channels + current_channel] += input[current_y * (width * channels) + current_x * channels + current_channel];
            }
        }
    }
}

void launch_add(const unsigned int* device_data_input,
    const unsigned int input_position, const unsigned int width, const unsigned int height, const unsigned int channels,
    unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
    const unsigned int type
) {
    cudaError_t err = cudaSuccess;

    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height * output_channels + threadsPerBlock - 1) / threadsPerBlock;

    addKernel<<<blocksPerGrid, threadsPerBlock>>> (device_data_input, input_position, width, height, channels, device_data_output, output_position, output_width, output_height, output_channels, type);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed in addKernel (error code %s)\n", cudaGetErrorString(err));
    }
}

vector3<unsigned int> bottleneck_get_max_size(struct bottleneck* b, struct vector2<unsigned int> input_dim) {
	vector3<unsigned int> max_output_dim;
	unsigned int max_output_size = 0;

	vector3<unsigned int> output_dim_0 = conv2d_get_output_dim(&b->conv1, input_dim);
	unsigned int output_size_0 = output_dim_0[0] * output_dim_0[1] * output_dim_0[2];
	b->conv1_output_dim = output_dim_0;
	max_output_dim = output_dim_0;
	max_output_size = output_size_0;

	vector3<unsigned int> output_dim_1 = conv2d_get_output_dim(&b->conv2, struct vector2<unsigned int>(output_dim_0[0], output_dim_0[1]));
	unsigned int output_size_1 = output_dim_1[0] * output_dim_1[1] * output_dim_1[2];
	if (output_size_1 > max_output_size) {
		max_output_size = output_size_1;
		max_output_dim = output_dim_1;
	}
	b->conv2_output_dim = output_dim_1;

	vector3<unsigned int> output_dim_2 = conv2d_get_output_dim(&b->conv3, struct vector2<unsigned int>(output_dim_1[0], output_dim_1[1]));
	unsigned int output_size_2 = output_dim_2[0] * output_dim_2[1] * output_dim_2[2];
	if (output_size_2 > max_output_size) {
		max_output_size = output_size_2;
		max_output_dim = output_dim_2;
	}
	b->conv3_output_dim = output_dim_2;

	return max_output_dim;
}