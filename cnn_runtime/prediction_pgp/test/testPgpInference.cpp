#include <iostream>
#include <string.h>

#include "pgp_net_infer.h"
#include "OrtSessionHandler.h"

    
int main()
{
    std::string model_path = "./config/model/pgp.onnx";
    Ort::PgpNetInference osh(model_path, 0);

    std::vector<float*> inputImgData;

    float* dst_input_1 = new float[1 * 5 * 5];
    memset(dst_input_1, 0.1, sizeof (float) * 1 * 5 * 5);
    inputImgData.push_back(dst_input_1);
    float* dst_input_7 = new float[1 * 164 * 20 * 6];
    memset(dst_input_7, 0.1, sizeof (float) * 1 * 164 * 20 * 6);
    inputImgData.push_back(dst_input_7);
    float* dst_lane_node_masks = new float[1 * 164 * 20 * 6];
    memset(dst_lane_node_masks, 0.1, sizeof (float) * 1 * 164 * 20 * 6);
    inputImgData.push_back(dst_lane_node_masks);
    float* dst_f4 = new float[1 * 84 * 5 * 5];
    memset(dst_f4, 0.1, sizeof (float) * 1 * 84 * 5 * 5);
    inputImgData.push_back(dst_f4);
    float* dst_nbr_vehicle_masks = new float[1 * 84 * 5 * 5];
    memset(dst_nbr_vehicle_masks, 0.1, sizeof (float) * 1 * 84 * 5 * 5);
    inputImgData.push_back(dst_nbr_vehicle_masks);
    float* dst_f6 = new float[1 * 77 * 5 * 5];
    memset(dst_f6, 0.1, sizeof (float) * 1 * 77 * 5 * 5);
    inputImgData.push_back(dst_f6);
    float* dst_nbr_ped_masks = new float[1 * 77 * 5 * 5];
    memset(dst_nbr_ped_masks, 0.1, sizeof (float) * 1 * 77 * 5 * 5);
    inputImgData.push_back(dst_nbr_ped_masks);
    float* dst_f8 = new float[1 * 164 * 84];
    memset(dst_f8, 0.1, sizeof (float) * 1 * 164 * 84);
    inputImgData.push_back(dst_f8);
    float* dst_f9 = new float[1 * 164 * 77];
    memset(dst_f9, 0.1, sizeof (float) * 1 * 164 * 77);
    inputImgData.push_back(dst_f9);
    float* dst_9 = new float[1 * 164 * 15];
    memset(dst_9, 0.1, sizeof (float) * 1 * 164 * 15);
    inputImgData.push_back(dst_9);
    float* dst_edge_type = new float[1 * 164 * 15];
    memset(dst_edge_type, 0.1, sizeof (float) * 1 * 164 * 15);
    inputImgData.push_back(dst_edge_type);

    auto inferenceOutput = osh(inputImgData);
    std::cout << inferenceOutput.size() << std::endl;
    // Ort::OrtSessionHandler osh(model_path, 0);
    // std::cout << osh::OrtSessionHandlerIml.GetInputNums() << std::endl;

    // float* dst = new float[IMG_WIDTH * IMG_HEIGHT * IMG_CHANNEL];
    // osh.preprocess(dst, img.data, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, Ort::IMAGENET_MEAN, Ort::IMAGENET_STD);

    // std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < TEST_TIMES; ++i) {
    //     auto inferenceOutput = osh({reinterpret_cast<float*>(dst)});

    //     const int TOP_K = 5;
    //     // osh.topK({inferenceOutput[0].first}, TOP_K);
    //     std::cout << osh.topKToString({inferenceOutput[0].first}, TOP_K) << std::endl;
    // }
    // std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    // auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    // std::cout << elapsedTime.count() / 1000. << "[sec]" << std::endl;

    // delete[] dst;
}