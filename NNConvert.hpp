#ifndef NNCONVERT_HPP
#define NNCONVERT_HPP

#include "BitField.hpp"
#include <string>
#include <map>

void read_into_bit_field_and_save(struct bit_field* bf, std::string path, std::string savepath_position, std::string savepath_data);
std::map<std::string, int> read_position_file(std::string filepath_position);

#endif