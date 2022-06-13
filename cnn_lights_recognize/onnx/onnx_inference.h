
#ifndef onnx_inference_
#define onnx_inference_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <fstream>

class OnnxInference
{
public:
    OnnxInference();
    ~OnnxInference();
    std::vector<float> onnx_pred(cv::Mat image,std::string onnx_path);

private:
    std::string onnx_path="/home/lpj/Desktop/easy_ml_inference/cnn_lights_recognize/ResNetCls.onnx";
    const std::string json_path="/home/lpj/Desktop/easy_ml_inference/cnn_lights_recognize/hyp.json";
    std::vector<double> mean={0.5070751592371323,0.48654887331495095,0.4409178433670343};
    double std=0.2666410733740041;
    int opencv_shape=256;
};

#endif /* onnx_inference_ */
