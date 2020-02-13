#ifndef CONV2D_HPP
#define CONV2D_HPP

#include "Vector2.hpp"
#include "Vector3.hpp"

struct conv2d {
	unsigned int					channels;
	unsigned int					filters;
	struct vector2<unsigned int>	kernel_size;
	struct vector2<unsigned int>	stride;
	struct vector2<unsigned int>	padding;
	struct vector2<unsigned int>	dilation;

	unsigned int					parameters_position;
};

void launch_conv2d(const unsigned int* device_data_parameters, const unsigned int parameters_position,
	const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h,
	unsigned int* device_data_output, const unsigned int output_position,
	struct conv2d* c);

void conv2d_init(struct conv2d* c, unsigned int channels, unsigned int filters, struct vector2<unsigned int> kernel_size, struct vector2<unsigned int> stride, struct vector2<unsigned int> padding, struct vector2<unsigned int> dilation, unsigned int parameters_position);
struct vector3<unsigned int> conv2d_get_output_dim(struct conv2d* c, struct vector2<unsigned int> input_dim);

#endif