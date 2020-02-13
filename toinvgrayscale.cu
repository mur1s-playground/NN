#include "toinvgrayscale.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

__global__ void toinvgrayscaleKernel(unsigned int* device_data,
    const unsigned int frame_position,
        const unsigned int width, const unsigned int height, const unsigned int channels,
    const unsigned int frame_position_target) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width * height) {
        unsigned char* frame = (unsigned char*)&device_data[frame_position];
        unsigned char* target_frame = (unsigned char*)&device_data[frame_position_target];

        int current_x = (i % width);
        int current_y = (i / width);

        float value = 0.0f;
        for (int c = 0; c < channels; c++) {
            value += frame[current_y * (width * channels) + current_x * channels + c];
        }
        target_frame[current_y * width + current_x] = 255 - (unsigned char)roundf(value / (float)channels);
    }
}

void launch_toinvgrayscale(unsigned int* device_data,
        const unsigned int frame_position,
        const unsigned int width, const unsigned int height, const unsigned int channels,
        const unsigned int frame_position_target) {
    
    cudaError_t err = cudaSuccess;

    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    toinvgrayscaleKernel<<<blocksPerGrid, threadsPerBlock>>> (device_data, frame_position, width, height, channels, frame_position_target);
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed in toinvgrayscaleKernel (error code %s)\n", cudaGetErrorString(err));
    }
}