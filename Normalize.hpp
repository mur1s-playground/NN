#ifndef NORMALIZE_HPP
#define NORMALIZE_HPP

#include "Vector3.hpp"

void launch_normalize(const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h, const unsigned int channels,
	unsigned int* device_data_output, const unsigned int output_position, const struct vector3<float> mean, const struct vector3<float> std);

#endif