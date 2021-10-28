#ifndef PATCHCORE_POSTPORCESS_H
#define PATCHCORE_POSTPORCESS_H

#include <opencv2/core/core.hpp>
#include <opencv2/ml.hpp>
#include <fstream>
#include <thread>
#include <atomic>
#include <sys/time.h>

#include "hnswlib.h"

unsigned long get_current_time(void);
cv::Mat train_embedding_process(const std::string &embedding_file, const int output_channel);

cv::Mat reshape_embedding(const float *output,
                          const int out_channel, \ 
                          const int out_height, \
                          const int out_width);

void knn_process(const cv::Mat &embedding_train, const cv::Mat &embedding_test, float* distances);



#endif // POESNET_POSTPORCESS_H