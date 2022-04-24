#ifndef _TRAFFIC_LIGHTS_CLASSIFIER_
#define _TRAFFIC_LIGHTS_CLASSIFIER_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "common/data_struct.h"
using namespace cv;
using namespace std;


class TrafficLightsClassifier
{
public:
    TrafficLightsClassifier();
    ~TrafficLightsClassifier();
    int estimate_label(int *result[4]);
    int * red_green_yellow(const cv::Mat &rgb_image);
    vector<Vec3f> hough_circles(Mat gray);
    int ** combine_circles(vector<Vec3f> circles,Mat image);
    vector<TrafficLightsParams> traffic_lights_result(Mat image,const std::vector<float> traffic_lights_locations);
    

public:
    vector<TrafficLightsParams> traffic_lights_results;
    // TrafficLightsParams traffic_lights_params;
    const vector<float> traffic_lights_locations={0,0,56,54};

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
    int shape=256;

};

#endif // _TRAFFIC_LIGHTS_CLASSIFIER_
