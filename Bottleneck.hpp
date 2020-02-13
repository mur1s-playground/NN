#ifndef BOTTLENECK_HPP
#define BOTTLENECK_HPP

#include <vector>

#include "Conv2d.hpp"
#include "BatchNorm2d.hpp"

#include "Vector2.hpp"
#include "Vector3.hpp"

struct bottleneck {
	struct conv2d		conv1;
	vector3<unsigned int> conv1_output_dim;
	struct bn2d			bn1;

	struct conv2d		conv2;
	vector3<unsigned int> conv2_output_dim;
	struct bn2d			bn2;
	
	struct conv2d		conv3;
	vector3<unsigned int> conv3_output_dim;
	struct bn2d			bn3;

	struct conv2d		conv_ds;
	struct bn2d			bn_ds;
};

void launch_add(const unsigned int* device_data_input,
	const unsigned int input_position, const unsigned int width, const unsigned int height, const unsigned int channels,
	unsigned int* device_data_output, const unsigned int output_position, const unsigned int output_width, const unsigned int output_height, const unsigned int output_channels,
	const unsigned int type
);

vector3<unsigned int> bottleneck_get_max_size(struct bottleneck* b, struct vector2<unsigned int> input_dim);

#endif