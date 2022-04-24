#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <unordered_map>
#include "traffic_lights_classifier.h"




using namespace std;
using namespace cv;

//
int main(int argc, char** argv)
{
    TrafficLightsClassifier traffic_lights_classifier;

    Mat image;
    image = imread("/Users/zhangzikai/Desktop/off.jpg");
    cout<<image.size<<endl;
    
    vector<TrafficLightsParams> result=traffic_lights_classifier.traffic_lights_result(image,traffic_lights_classifier.traffic_lights_locations);
    
    cout<<result[0].target_id<<endl;
    cout<<result[0].traffic_lights_type<<endl;
    cout<<result[0].traffic_lights_location[0][0]<<endl;
    cout<<result[0].traffic_lights_location[0][1]<<endl;
    cout<<result[0].traffic_lights_location[0][2]<<endl;
    cout<<result[0].traffic_lights_location[0][3]<<endl;

    return 0;
}
