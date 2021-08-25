#include <iostream>
#include <fstream>
#include <math.h>
//opencv
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#define KNEIGHBOURS (9)
#define LENGTH (2515968)

void reshape_embedding(const cv::Mat &embedding_array, cv::Mat embedding_test)
{
    int out_channel = embedding_array.size[1];
    int out_height = embedding_array.size[2];
    int out_width = embedding_array.size[3];
    std::cout << out_channel << " " << out_height << " " << out_width << std::endl;
    for (int h = 0; h < out_height; h++)
    {
        for (int w = 0; w < out_width; w++)
        {
            for (int c = 0; c < out_channel; c++)
            {
                ((float*)embedding_test.data)[h*out_width*out_channel + w*out_channel + c] = 
                    ((float*)embedding_array.data)[c*out_height*out_height + h*out_width + w];
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

float postprocess(const cv::Mat &embedding_coreset, const cv::Mat &embedding_test)
{
    const int K(9);
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    knn->setDefaultK(K);
    knn->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
    cv::Mat labels(embedding_coreset.rows, 1, CV_32FC1, cv::Scalar(0.0));

    knn->train(embedding_coreset, cv::ml::ROW_SAMPLE, labels);

    cv::Mat result, neighborResponses, distances_a;
    knn->findNearest(embedding_test, K, result, neighborResponses, distances_a);
    // std::cout << "result: " << distances.size[0] << " " << distances.size[1] << std::endl;

    float* distances = new float[784*9];
    memcpy(distances, distances_a.data, sizeof(float)*784*9);

    // float* distances_numpy = new float[784*9];
    // std::ifstream distance_f;
    // distance_f.open("/home/edge/easy_data/easy_ml_inference/cnn_runtime/one_class/distances.bin", std::ifstream::binary);
    // distance_f.read(reinterpret_cast<char*>(distances_numpy), sizeof(float) * 784 * 9);
    // distance_f.close();

    // float* distances = new float[784*9];
    // for (int d = 0; d < 9; d++)
    // {
    //     for (int c = 0; c < 784; c++)
    //     {
    //         memcpy(distances + d*784 + c, distances_numpy + c*9 + d, sizeof(float));
    //     }
    // }

    // for (int b = 0; b < 20; b++)
    // {
    //     std::cout << distances[b] << " ";
    // }

    // float* max_neighbours = new float[28*28];
    // memcpy(max_neighbours, distances + 28 * 28, sizeof(float)*28*28);
    int max_posit = std::max_element(distances, \
				                distances + 28 * 28) - distances; // - distances.data;

    std::cout << "max_posit: " << max_posit << std::endl;

    float* N_b = new float[9];
    for(int i = 0; i < 9; i++)
    {
        N_b[i] = distances[i * 28 * 28 + max_posit];
        // N_b[i] = ((float*)distances.data)[i * 28 * 28 + max_posit];
    }
    float w, sum_N_b = 0;
    for(int j = 0; j < 9; j++)
    {
        sum_N_b += exp(N_b[j]);
    }
    float max_N_b = *std::max_element(N_b, N_b + 9);
    w = (1 - exp(max_N_b) / sum_N_b);
    float score;
    // score = w * ((float*)distances.data)[max_posit];
    score = w * distances[max_posit];

    return score;
}

int main()
{
    std::string model_file = "/home/edge/easy_data/easy_ml_inference/cnn_runtime/one_class/OneClassNet.onnx";
    std::string image_file = "/home/edge/easy_data/easy_ml_inference/cnn_runtime/one_class/000.png";
    //"/easy_data/easy_ml_inference/cnn_runtime/one_class/000.png";
    std::string image_txt_path = "/home/edge/data/VOCdevkit/MVtec/bottle/ImageSets/val.txt";
    std::string image_dir = "/home/edge/data/VOCdevkit/MVtec/bottle/JPEGImages/";
    std::string embedding_file = "/home/edge/easy_data/easy_ml_inference/cnn_runtime/one_class/embedding.bin";

    float* embedding_coreset = new float[LENGTH];
    std::ifstream embedding;
    embedding.open(embedding_file, std::ifstream::binary);
    embedding.read(reinterpret_cast<char*>(embedding_coreset), sizeof(float) * LENGTH);
    embedding.close();

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
        // for (int c = 0; c < 20; c++)
        // {
        //     std::cout << ((float*)out.data)[c] << " ";
        // }
        // std::cout << std::endl;

        float score;
        cv::Mat embedding_train(LENGTH/out_channel, out_channel, CV_32FC1);
        cv::Mat embedding_test(out_height*out_width, out_channel, CV_32FC1);

        memcpy(embedding_train.data, embedding_coreset, LENGTH*sizeof(float));
        reshape_embedding(out, embedding_test);

        // for (int c = 0; c < 5; c++)
        // {
        //     std::cout << embedding_train.at<float>(c, 0) << " ";
        // }
        // std::cout << std::endl;
        // for (int c = 0; c < 20; c++)
        // {
        //     std::cout << ((float*)embedding_train.data)[c] << " ";
        // }
        // std::cout << std::endl;
        // for (int c = 0; c < 20; c++)
        // {
        //     std::cout << ((float*)embedding_test.data)[c] << " ";
        // }
        // std::cout << std::endl;

        // float* embedding_numpy = new float[out_height*out_width*out_channel];
        // std::ifstream embedding_f;
        // embedding_f.open("/home/edge/easy_data/easy_ml_inference/cnn_runtime/one_class/embedding_test.bin", std::ifstream::binary);
        // embedding_f.read(reinterpret_cast<char*>(embedding_numpy), sizeof(float)*out_height*out_width*out_channel);
        // embedding_f.close();

        // for (int b = 0; b < 20; b++)
        // {
        //     std::cout << embedding_coreset[b] << " ";
        // }
        // memcpy(embedding_test.data, embedding_numpy, sizeof(float)*out_height*out_width*out_channel);

        score = postprocess(embedding_train, embedding_test);
        std::cout << "score: " << score << std::endl;
    }

    return 0;
}