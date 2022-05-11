#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "traffic_lights_classifier.h"
#include "json/json.h"
#include <fstream>


//class ONNXClassifier
//{
//public:
//    ONNXClassifier(const std::string& model_path, const std::string& label_path, cv::Size _input_size);
//    void Classify(const cv::Mat& input_image, std::string& out_name);
//private:
//    void preprocess_input(cv::Mat& image);
//    bool read_labels(const std::string& label_paht);
//private:
//    cv::Size input_size;
//    cv::dnn::Net net;
//    cv::Scalar default_mean;
//    cv::Scalar default_std;
//    std::vector<std::string> labels;
//
//};
//
//ONNXClassifier::ONNXClassifier(const std::string& model_path, const std::string& label_path, cv::Size _input_size):default_mean(0.485, 0.456, 0.406),
//default_std(0.229, 0.224, 0.225),input_size(_input_size)
//{
//    if (!read_labels(label_path))
//    {
//        throw std::runtime_error("label read fail!");
//    }
//    net = cv::dnn::readNet(model_path);
//    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
//    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
//}
//bool ONNXClassifier::read_labels(const std::string& label_path)
//{
//    std::ifstream ifs(label_path);
//    assert(ifs.is_open());
//    std::string line;
//    while (std::getline(ifs,line))
//    {
//        std::size_t index = line.find_first_of(':');
//        labels.push_back(line.substr(index + 1));
//    }
//    if (labels.size() > 0)
//        return true;
//    else
//        return false;
//}
//void ONNXClassifier::preprocess_input(cv::Mat& image)
//{
//    image.convertTo(image, CV_32F,1.0/255.0);
//    cv::subtract(image,default_mean,image);
//    cv::divide(image, default_std, image);
//}
//
//void ONNXClassifier::Classify(const cv::Mat& input_image, std::string& out_name)
//{
//    out_name.clear();
//    cv::Mat image = input_image.clone();
//    preprocess_input(image);
//    cv::Mat input_blob = cv::dnn::blobFromImage(image, 1.0, input_size, cv::Scalar(0, 0, 0), true);
//    net.setInput(input_blob);
//    const std::vector<cv::String>& out_names = net.getUnconnectedOutLayersNames();
//    cv::Mat out_tensor = net.forward(out_names[0]);
//        cv::Point maxLoc;
//        cv::minMaxLoc(out_tensor,(double*)0,(double*)0,(cv::Point*)0,&maxLoc);
//        out_name = labels[maxLoc.x];
//}
//
//int main(int argc, char* argv[])
//{
////    if (argc != 2)
////    {
////        std::cout << "input a image file path" << std::endl;
////        return -1;
////    }
//    std::string model_path("/Users/zhangzikai/Downloads/ResNetCls.onnx");
//    std::string label_path("/Users/zhangzikai/Desktop/labels.txt");
//    cv::Size input_size(224, 224);
//    cv::Mat test_image = cv::imread("/Users/zhangzikai/Desktop/yellow.jpg");
//    ONNXClassifier classifier(model_path, label_path, input_size);
//    std::string result;
//    classifier.Classify(test_image, result);
//        std::cout<<"result: "<<result<<std::endl;
//    return 0;
//}


int main(int argc, char** argv)
{
    TrafficLightsClassifier traffic_lights_classifier;
//
   cv::Mat image;
   image = cv::imread("/Users/zhangzikai/Desktop/off.jpg");
   std::cout<<image.size<<std::endl;

   vector<TrafficLightsParams> result=traffic_lights_classifier.traffic_lights_result(image,traffic_lights_classifier.traffic_lights_locations);
//
   std::cout<<result[0].target_id<<std::endl;
   std::cout<<result[0].traffic_lights_type<<std::endl;
   std::cout<<result[0].traffic_lights_location[0][0]<<std::endl;
   std::cout<<result[0].traffic_lights_location[0][1]<<std::endl;
   std::cout<<result[0].traffic_lights_location[0][2]<<std::endl;
   std::cout<<result[0].traffic_lights_location[0][3]<<std::endl;
    
//    json j;
//    j={{"low_green",10},
//        {"up_green",50},
//        {"low_yellow",100},
//        {"up_yellow",115},
//        {"low_red",116},
//        {"up_red",130},
//        {"low_off",0},
//        {"up_off",10},
//        {"shape",256}
//    };
//
//
    
    
   return 0;
}
