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
    int estimate_label(int *result[4]);
    int * red_green_yellow(const cv::Mat &rgb_image);
    vector<cv::Vec3f> hough_circles(cv::Mat gray);
    int ** combine_circles(vector<cv::Vec3f> circles,cv::Mat image);
    vector<TrafficLightsParams> traffic_lights_result(cv::Mat image,const std::vector<float> traffic_lights_locations);
    

public:
    vector<TrafficLightsParams> traffic_lights_results;
    // TrafficLightsParams traffic_lights_params;
    const vector<float> traffic_lights_locations={0,0,56,54};

private:
    // void red_green_yellow(cv::Mat rgb_image);
    int low_green;
    int up_green;
    int low_yellow;
    int up_yellow;
    int low_red;
    int up_red;
    int low_off;
    int up_off;
    int shape;

};

#endif // _TRAFFIC_LIGHTS_CLASSIFIER_
