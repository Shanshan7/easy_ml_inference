
#include "onnx_inference.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "json/json.h"
#include <fstream>
#include <numeric>


OnnxInference::OnnxInference()
{

    std::ifstream in(json_path, std::ios::binary);
    Json::Reader reader;
    Json::Value root;
//
    if(reader.parse(in, root))
    {
        onnx_path=root["onnx_path"].asString();
        opencv_shape=root["opencv_shape"].asInt();
    }


    else
    {
        std::cout << "Error opening file\n";
        exit(0);
    }

    in.close();
}

OnnxInference:: ~OnnxInference()
{

}

std::vector<float> OnnxInference::onnx_pred(cv::Mat image,std::string onnx_path){
    cv::dnn::Net net = cv::dnn::readNetFromONNX(onnx_path);
    cv::Mat blob = cv::dnn::blobFromImage(image,1/(std*255),cv::Size(opencv_shape,opencv_shape),cv::Scalar(mean[0],mean[1],mean[2])*255,true,false);  // 由图片加载数据 这里还可以进行缩放、归一化等预处理
    net.setInput(blob);  // 设置模型输入
    cv::Mat predict = net.forward(); // 推理出结果
    return std::vector<float>(predict);
}
