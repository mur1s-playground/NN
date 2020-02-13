#ifndef MAXPOOL2D_HPP
#define MAXPOOL2D_HPP

#include "Vector2.hpp"
#include "Vector3.hpp"

struct maxpool2d {
	unsigned int					filters;
	struct vector2<unsigned int>	kernel_size;
	struct vector2<unsigned int>	stride;
	struct vector2<unsigned int>	padding;
	struct vector2<unsigned int>	dilation;
};

void launch_maxpool2d(
	const unsigned int* device_data_input, const unsigned int input_position, const unsigned int input_w, const unsigned int input_h,
	unsigned int* device_data_output, const unsigned int output_position,
	struct maxpool2d* m);

void maxpool2d_init(struct maxpool2d* m, unsigned int filters, struct vector2<unsigned int> kernel_size, struct vector2<unsigned int> stride, struct vector2<unsigned int> padding, struct vector2<unsigned int> dilation);

struct vector3<unsigned int> maxpool2d_get_output_dim(struct maxpool2d* m, struct vector2<unsigned int> input_dim);

#endif