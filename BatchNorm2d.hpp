#ifndef BATCHNORM2D_HPP
#define BATCHNORM2D_HPP

struct bn2d {
	unsigned int filters;

	unsigned int position_weights;
	unsigned int position_bias;
};

void launch_batchnorm2d(const unsigned int* device_data_parameters, const unsigned int parameters_position_weights, const unsigned int parameters_position_bias,
	const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int filters);


void batchnorm2d_init(struct bn2d* bn, const unsigned int filters, const unsigned int position_weights, const unsigned int position_bias);
#endif