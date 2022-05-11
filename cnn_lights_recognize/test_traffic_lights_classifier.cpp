#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "traffic_lights_classifier.h"
#include "json/json.h"
#include <fstream>




//
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
