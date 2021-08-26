#include <iostream>
#include <fstream>
#include <math.h>
//opencv
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include <sys/time.h>

#define KNEIGHBOURS (9)
// #define FEATUREWIDTH (1638)
#define FEATUREHEIGHT (1536)

unsigned long get_current_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec*1000000 + tv.tv_usec);
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

void preprocess(const cv::Mat &src_mat, const cv::Size dst_size, cv::Mat &dst_image)
{
    if(src_mat.empty()){
        return;
    }

    cv::Mat resize_mat;
    cv::resize(src_mat, resize_mat, dst_size, 0, 0, cv::INTER_LINEAR);

    std::vector<float> mean_value{123.675,116.28,103.53};
    std::vector<float> std_value{57.63,57.63,57.63};
    std::vector<cv::Mat> src_channels(3);
    cv::split(resize_mat, src_channels);

    for (int i = 0; i < src_channels.size(); i++)
    {
        src_channels[i].convertTo(src_channels[i], CV_32FC1);
        src_channels[i] = (src_channels[i] - mean_value[i]) / (0.00001 + std_value[i]);
    }
    cv::merge(src_channels, dst_image);
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

    for (int e = 0; e < 20; e++)
    {
        std::cout << output[e] << " ";
    }
    std::cout << "\n";

    cv::Mat embedding_train(embedding_length / FEATUREHEIGHT, FEATUREHEIGHT, CV_32FC1);
    cv::Mat embedding_test(out_height*out_width, out_channel, CV_32FC1);

    memcpy(embedding_train.data, embedding_coreset, embedding_length * sizeof(float));
    reshape_embedding(output, embedding_test, out_channel, out_height, out_width);

    for (int a = 0; a < 20; a++)
    {
        std::cout << ((float*)embedding_train.data)[a] << " ";
    }
    std::cout << "\n";
    for (int a = 0; a < 20; a++)
    {
        std::cout << ((float*)embedding_test.data)[a] << " ";
    }
    std::cout << "\n";

    //----------------------------knn---------------------------
    const int K(KNEIGHBOURS);
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    knn->setDefaultK(K);
    knn->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
    cv::Mat labels(embedding_train.rows, 1, CV_32FC1, cv::Scalar(0.0));

    knn->train(embedding_train, cv::ml::ROW_SAMPLE, labels);

    cv::Mat result, neighborResponses, distances_mat;
    knn->findNearest(embedding_test, K, result, neighborResponses, distances_mat);

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