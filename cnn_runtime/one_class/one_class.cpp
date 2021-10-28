#include <iostream>
#include <fstream>
#include <math.h>
//opencv
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include "patchcore_postprocess.h"

#include <sys/time.h>

#define KNEIGHBOURS (9)

float postprocess(const float *output, \ 
                  const std::string &embedding_file, \ 
                  const int out_channel, \ 
                  const int out_height, \
                  const int out_width)
{
    cv::Mat embedding_train = train_embedding_process(embedding_file, out_channel);
    cv::Mat embedding_test = reshape_embedding(output, out_channel, out_height, out_width);
    float* distances = new float[embedding_test.rows*KNEIGHBOURS];
    knn_process(embedding_train, embedding_test, distances);

    int max_posit = std::max_element(distances, \
				                distances + embedding_test.rows) - distances; // - distances.data;

    // std::cout << "max_posit: " << max_posit << std::endl;

    float* N_b = new float[KNEIGHBOURS];
    for(int i = 0; i < KNEIGHBOURS; i++)
    {
        N_b[i] = distances[i * embedding_test.rows + max_posit];
    }

    float w, sum_N_b = 0;
    for(int j = 0; j < KNEIGHBOURS; j++)
    {
        sum_N_b += exp(N_b[j]);
    }
    float max_N_b = *std::max_element(N_b, N_b + KNEIGHBOURS);
    w = (1 - exp(max_N_b) / sum_N_b);

    float score;
    score = w * distances[max_posit];

    return score;
}

int main()
{
    std::string model_file = "/home/edge/easy_data/easy_ml_inference/cnn_runtime/one_class/OneClassNet.onnx";
    std::string image_txt_path = "/home/edge/data/VOCdevkit/MVtec/grid/ImageSets/val.txt";
    std::string image_dir = "/home/edge/data/VOCdevkit/MVtec/grid/JPEGImages/";
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