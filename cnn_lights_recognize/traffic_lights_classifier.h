#ifndef _TRAFFIC_LIGHTS_CLASSIFIER_
#define _TRAFFIC_LIGHTS_CLASSIFIER_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <fstream>

#include "common/data_struct.h"




class TrafficLightsClassifier
{
public:
    TrafficLightsClassifier();
    ~TrafficLightsClassifier();
    vector<float> estimate_label(int *result[4]);
    int * red_green_yellow(const cv::Mat &rgb_image);
    vector<cv::Vec3f> hough_circles(cv::Mat gray);
    int ** combine_circles(vector<cv::Vec3f> circles,cv::Mat image);
    vector<TrafficLightsParams> traffic_lights_result(cv::Mat image,const std::vector<float> traffic_lights_locations,bool onnx=true);
    vector<float> onnx_pred(cv::Mat image,string onnx_path);
    

public:
    vector<TrafficLightsParams> traffic_lights_results;
    // TrafficLightsParams traffic_lights_params;
    const vector<float> traffic_lights_locations={0,0,21,43};

private:
    // void red_green_yellow(cv::Mat rgb_image);
    int low_green=10;
    int up_green=50;
    int low_yellow=100;
    int up_yellow=115;
    int low_red=116;
    int up_red=130;
    int low_off=0;
    int up_off=10;
    int shape=224;
    double w1=0.5;
    double w2=0.5;
    string onnx_path="/Users/zhangzikai/Downloads/ResNetCls.onnx";
    vector<double> mean={0.5070751592371323,0.48654887331495095,0.4409178433670343};
    double std=0.2666410733740041;
};

#endif // _TRAFFIC_LIGHTS_CLASSIFIER_
