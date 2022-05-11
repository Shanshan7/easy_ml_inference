#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <dirent.h>
#include <sys/time.h>
#include <thread>
#include <mutex>
#include <future>


unsigned long get_current_time(void);

void ListPath(std::string const &path, std::vector<std::string> &paths);

void ListImages(std::string const &path, std::vector<std::string> &images);

std::vector<std::vector<float>> martrix_multiply_num(std::vector<std::vector<float>> matrix, float num);
void array_reverse(float* array, int array_length);

bool copy_file(std::string src, std::string dest);
void conv1d(float *input, float *kernel, int input_length, int kernel_size, float *output);
void conv1d(std::vector<float> &input, float *kernel, int input_length, int kernel_size, float *output);

#endif //__UTILS_H__