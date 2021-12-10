#include <iostream>
#include <fstream>
#include <math.h>
//opencv
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <hnswlib.h>
#include <thread>

#include <sys/time.h>
#include "patchcore_postprocess.h"

#define KNEIGHBOURS (9)

unsigned long get_current_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec*1000000 + tv.tv_usec);
}

template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if ((id >= end)) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

void reshape_embedding(const float *output, \
                       cv::Mat embedding_test, \
                       const int out_channel, \ 
                       const int out_height, \
                       const int out_width)
{
    for (int h = 0; h < out_height; h++)
    {
        for (int w = 0; w < out_width; w++)
        {
            for (int c = 0; c < out_channel; c++)
            {
                ((float*)embedding_test.data)[h*out_width*out_channel + w*out_channel + c] = 
                    output[c*out_height*out_height + h*out_width + w];
                // memcpy(embedding_test.data + h*out_width*out_channel + w*out_channel + c, \
                // embedding_array.data + c*out_height*out_height + h*out_width + w, sizeof(float));
            }
        }
    }
}

float postprocess(const float *output, \ 
                  const std::string &embedding_file, \ 
                  const int out_channel, \ 
                  const int out_height, \
                  const int out_width)
{
    // float* embedding_coreset = new float[FEATUREWIDTH * FEATUREHEIGHT];
    // std::ifstream embedding;
    // embedding.open(embedding_file, std::ifstream::binary);
    // // embedding.seekg(0, std::ios::end);
    // // int len = embedding.tellg();
    // // std::cout << len / sizeof(float) << std::endl;
    // embedding.read(reinterpret_cast<char*>(embedding_coreset), sizeof(float) * FEATUREWIDTH * FEATUREHEIGHT);
    // embedding.close();

    std::ifstream embedding(embedding_file, std::ios::binary|std::ios::in);
    embedding.seekg(0,std::ios::end);
    int embedding_length = embedding.tellg() / sizeof(float);
    embedding.seekg(0, std::ios::beg);
    float* embedding_coreset = new float[embedding_length];
    embedding.read(reinterpret_cast<char*>(embedding_coreset), sizeof(float) * embedding_length);
    embedding.close();

    // for (int e = 0; e < 20; e++)
    // {
    //     std::cout << output[e] << " ";
    // }
    // std::cout << "\n";

    cv::Mat embedding_train(embedding_length / out_channel, out_channel, CV_32FC1);
    cv::Mat embedding_test(out_height*out_width, out_channel, CV_32FC1);

    memcpy(embedding_train.data, embedding_coreset, embedding_length * sizeof(float));
    reshape_embedding(output, embedding_test, out_channel, out_height, out_width);
    cv::Mat distances_mat = knn_process(embedding_train, embedding_test);

    int distanceMatWidth = distances_mat.size[0];
    int distanceMatHeight = distances_mat.size[1];
    // std::cout << "result: " << distances_mat.size[0] << " " << distances_mat.size[1] << std::endl;

    // reshape 784 * 9 --> 9 * 784
    float* distances = new float[distanceMatWidth*distanceMatHeight];
    for (int d = 0; d < distanceMatHeight; d++)
    {
        for (int c = 0; c < distanceMatWidth; c++)
        {
            distances[d*distanceMatWidth + c] = ((float*)distances_mat.data)[c*distanceMatHeight + d];
            // memcpy(distances + d*784 + c, distances_mat.data + c*9 + d, sizeof(float));
        }
    }
    // //----------------------------------------------------------------------------

    // //----------------------------ann---------------------------------------------
    // const int K(KNEIGHBOURS);
    // int distanceMatWidth = embedding_test.rows;
    // int distanceMatHeight = K;
    // float* distances = new float[distanceMatWidth*distanceMatHeight];

    // int ef_construction = K * 10;
    // int M = 16;
    // // Declaring index，声明索引类型，如：l2, cosine or ip
    // auto space = std::unique_ptr<hnswlib::L2Space>(new hnswlib::L2Space(embedding_train.rows));
    // // std::cout << embedding_train.rows << " " << embedding_test.rows << std::endl;
    // // 初始化索引，元素的最大数需要是已知的
    // auto appr_alg = std::unique_ptr<hnswlib::HierarchicalNSW<float>>(
    //         new hnswlib::HierarchicalNSW<float>(space.get(), embedding_train.rows, M,
    //                                             ef_construction));
    // appr_alg->setEf(ef_construction);
    // // std::priority_queue<std::pair<float, hnswlib::labeltype >> ret=appr_alg->searchKnn(embedding_test.data, K);//'1' is the nearest neg
    // int count = 0;
    // int vecdim = embedding_train.cols;
    // unsigned long time_start_insert, time_end_insert, time_start_search, time_end_search;
    // time_start_insert = get_current_time();
    // // for (size_t i = 0; i < embedding_train.rows; i++)
    // // {
    // //     float *mass = new float[vecdim];
    // //     memcpy(mass, embedding_train.data + i*vecdim, vecdim * sizeof(float));
    // //     appr_alg->addPoint((void*)mass, count);
    // //     ++count;
    // //     delete[] mass;
    // // }
    // ParallelFor(0, embedding_train.rows, 16, [&](size_t row, size_t threadId)
    // {
    //     float *mass = new float[vecdim];
    //     memcpy(mass, embedding_train.data + row*vecdim, vecdim * sizeof(float));
    //     appr_alg->addPoint((void*)mass, row);
    //     delete[] mass;
    // });
    // time_end_insert = get_current_time();
    // std::cout << "addpoint cost time: " <<  (time_end_insert - time_start_insert)/1000.0  << "ms" << std::endl;

    // time_start_search = get_current_time();
	// // for (int i = 0; i < embedding_test.rows; i++)
	// // {
	// // 	float *mass = new float[vecdim];
    // //     memcpy(mass, embedding_test.data + i*vecdim, vecdim * sizeof(float));
	// // 	std::priority_queue<std::pair<float, hnswlib::labeltype>> candidates = 
    // //                                                     appr_alg->searchKnn((void*)mass, K);
	// // 	delete[] mass;

    // //     for (int k = 0; k < K; k++)
    // //     {
    // //         // std::cout << "i: " << i << ", k: " << k << ", dist: " << candidates.top().first << std::endl;
    // //         distances[i*K + k] = candidates.top().first;
    // //         candidates.pop();
    // //     }
	// // }
	// ParallelFor(0, embedding_test.rows, 16, [&](size_t row, size_t threadId)
	// {
	// 	float *mass = new float[vecdim];
    //     memcpy(mass, embedding_test.data + row*vecdim, vecdim * sizeof(float));
	// 	std::priority_queue<std::pair<float, hnswlib::labeltype>> candidates = 
    //                                                     appr_alg->searchKnn((void*)mass, K);
	// 	delete[] mass;

    //     for (int k = 0; k < K; k++)
    //     {
    //         // std::cout << "i: " << i << ", k: " << k << ", dist: " << candidates.top().first << std::endl;
    //         distances[row*K + k] = candidates.top().first;
    //         candidates.pop();
    //     }
	// });
    // time_end_search = get_current_time();
    // std::cout << "search cost time: " <<  (time_end_search - time_start_search)/1000.0  << "ms" << std::endl;
    // //-----------------------------------------------------------------------------

    int max_posit = std::max_element(distances, \
				                distances + distanceMatWidth) - distances; // - distances.data;

    // std::cout << "max_posit: " << max_posit << std::endl;

    float* N_b = new float[distanceMatHeight];
    for(int i = 0; i < distanceMatHeight; i++)
    {
        N_b[i] = distances[i * distanceMatWidth + max_posit];
    }

    float w, sum_N_b = 0;
    for(int j = 0; j < distanceMatHeight; j++)
    {
        sum_N_b += exp(N_b[j]);
    }
    float max_N_b = *std::max_element(N_b, N_b + distanceMatHeight);
    w = (1 - exp(max_N_b) / sum_N_b);

    float score;
    score = w * distances[max_posit];

    return score;
}

int main()
{
    std::string model_file = "/home/edge/easy_data/easy_ml_inference/cnn_runtime/one_class/OneClassNet.onnx";
    std::string image_txt_path = "/home/edge/data/VOCdevkit/MVtec/bottle/ImageSets/val.txt";
    std::string image_dir = "/home/edge/data/VOCdevkit/MVtec/bottle/JPEGImages/";
    std::string embedding_file = "/home/edge/easy_data/easy_ml_inference/cnn_runtime/one_class/embedding.bin";

    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_file);

    float scale = 1./57.63;
    cv::Scalar mean = cv::Scalar(123.675,116.28,103.53);

    std::ifstream read_txt;
    std::string line_data;
    read_txt.open(image_txt_path.data());
    if(!read_txt.is_open()){
        std::cout << image_txt_path << " not exits" << std::endl;
        return -1;
    }
    
    while(std::getline(read_txt, line_data))
    {
        if(line_data.empty()){
            continue;
        }
        size_t index = line_data.find_first_of(' ', 0);
        std::string image_name = line_data.substr(0, index);
        std::stringstream image_path;
        image_path << image_dir << image_name;
        std::cout << image_path.str() << std::endl;
        cv::Mat image = cv::imread(image_path.str());
        cv::Mat blob = cv::dnn::blobFromImage(image, scale, cv::Size(224, 224), mean, true);
        net.setInput(blob, "one_class_input");

        cv::Mat out = net.forward("one_class_output");

        int out_channel = out.size[1];
        int out_height = out.size[2];
        int out_width = out.size[3];
   
        float* output = new float[out_channel * out_height * out_width];
        memcpy(output, out.data, sizeof(float) * out_channel * out_height * out_width);

        float score;
        unsigned long time_start, time_end;
        time_start = get_current_time();
        score = postprocess(output, embedding_file, out_channel, out_height, out_width);
        time_end = get_current_time();
        std::cout << "one_class_net cost time: " <<  (time_end - time_start)/1000.0  << "ms" << std::endl;
        std::cout << "score: " << score << std::endl;
    }

    return 0;
}