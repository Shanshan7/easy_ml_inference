#include <iostream>
//opencv
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main()
{
    std::string model_file = "/easy_data/one_class/classnet.onnx";
    std::string image_file = "/easy_data/one_class/000.png";
    cv::dnn::Net net = cv::dnn::readNetFromONNX(model_file);
    cv::Mat image = cv::imread(image_file);
    cv::Mat blob = cv::dnn::blobFromImage(image);
    net.setInput(blob);

    cv::Mat out = net.forward();
    std::cout << "out channel: " << out.channels() << " out height: " << out.rows \
              << " out width: " << out.cols << std::endl;
    std::cout << out << std::endl;
    

    return 0;
}