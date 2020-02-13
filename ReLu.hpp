#ifndef RELU_HPP
#define RELU_HPP

void launch_relu(
	const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int filters);

#endif