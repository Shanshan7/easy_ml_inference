#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common/data_struct.h"


class TrafficLightsClassifier
{
public:
    TrafficLightsClassifier();
    ~TrafficLightsClassifier();
    void estimate_label();
    void red_green_yellow(const cv::Mat &rgb_image, const std::vector<float> traffic_lights_locations);

public:
    std::vector<TrafficLightsParams> traffic_lights_results;
    // TrafficLightsParams traffic_lights_params;

private:
    // void red_green_yellow(cv::Mat rgb_image);
};