#include "NNConvert.hpp"

#include <windows.h>
#include <tchar.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

vector<string> get_all_files_names_within_folder(string folder) {
    vector<string> names;
    string search_path = folder + "/*.*";
    WIN32_FIND_DATA fd;
    HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                names.push_back(fd.cFileName);
            }
        } while (::FindNextFile(hFind, &fd));
        ::FindClose(hFind);
    }
    return names;
}

std::string& ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(0, str.find_first_not_of(chars));
    return str;
}

std::string& rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(str.find_last_not_of(chars) + 1);
    return str;
}

std::string& trim(std::string& str, const std::string& chars = " ")
{
    return ltrim(rtrim(str, chars), chars);
}

void read_into_bit_field_and_save(struct bit_field *bf, string path, string savepath_position, string savepath_data) {
    vector<string> files = get_all_files_names_within_folder(path);
    ofstream position_file;
    position_file.open(savepath_position);

    for (int i = 0; i < files.size(); i++) {
        string filepath = path + "/" + files[i];
        printf("processing file: %s : ", files[i].c_str());

        vector<float> numbers;
        numbers.reserve(9000000);

        std::ifstream file(filepath);
        std::string filecontent;
        int counter = 0;
        if (file.is_open()) {
            file.clear();
            file.seekg(0, ios::beg);
            while (std::getline(file, filecontent)) {
                size_t last_pos = 0;
                size_t pos = 0;
                std::string token;
                filecontent += "\n";
                while ((pos = filecontent.find(' ', last_pos)) != std::string::npos) {
                    token = filecontent.substr(last_pos, pos);
                    token = trim(token);

                        try {
                            float tmp = stof(token);
                            numbers.insert(numbers.end(), tmp);
                            counter++;
                        } catch (invalid_argument) {
                        }
                        last_pos = pos + 1;
                }
            }
        }

        
        printf("%I64u\n", numbers.size());
        int size = numbers.size() * sizeof(float);
        int size_in_bf = (int)ceil(size/(float)sizeof(unsigned int));
        int position = bit_field_add_bulk(bf, (unsigned int *) numbers.data(), size_in_bf, size);
        position_file << files[i] << ":" << position << " ";
    }
    position_file.close();
    bit_field_save_to_disk(bf, savepath_data);
}

std::map<std::string, int> read_position_file(std::string filepath_position) {
    std::ifstream t(filepath_position);
    std::stringstream buffer;
    buffer << t.rdbuf();
    t.close();
    int pos = 0;
    std::string filecontent = buffer.str();
    std::map<std::string, int> map;
    while ((pos = filecontent.find(" ", 0)) != std::string::npos) {
        std::string token = filecontent.substr(0, pos);
        if (token.size() <= 1) break;
        int delp = token.find(":");
        std::string name = token.substr(0, delp);
        int position = stoi(token.substr(delp+1));
        map.insert(std::pair<std::string, int>(name, position));
        filecontent.erase(0, pos+1);
    }
    return map;
}