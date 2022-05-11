#include "traffic_lights_classifier.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "json/json.h"
#include <fstream>


TrafficLightsClassifier::TrafficLightsClassifier()
{
    traffic_lights_results.clear();
    
    std::ifstream in("/Users/zhangzikai/Desktop/easy_ml_inference/cnn_lights_recognize/hyp.json", ios::binary);
    Json::Reader reader;
    Json::Value root;
    
    if(reader.parse(in, root))
    {
        low_green=root["low_green"].asInt();
        up_green=root["up_green"].asInt();
        low_yellow=root["low_yellow"].asInt();
        up_yellow=root["up_yellow"].asInt();
        low_red=root["low_red"].asInt();
        up_red=root["up_red"].asInt();
        low_off=root["low_off"].asInt();
        up_off=root["up_off"].asInt();
        shape=root["shape"].asInt();
    
    }
    
    
    else
    {
        std::cout << "Error opening file\n";
        exit(0);
    }
    
    in.close();
}

TrafficLightsClassifier:: ~TrafficLightsClassifier()
{

}

int * TrafficLightsClassifier::red_green_yellow(const cv::Mat &rgb_image)
{
    int *M=(int *)malloc(4*sizeof(int));
    
    cv::Mat resize_rgb_image, hsv_image;
//    rgb_image_roi = rgb_image(cv::Rect(traffic_lights_locations[0], traffic_lights_locations[1], traffic_lights_locations[2], \
//                                        traffic_lights_locations[3]));
    cv::resize(rgb_image, resize_rgb_image, cv::Size(shape, shape));
    cv::cvtColor(resize_rgb_image, hsv_image, cv::COLOR_RGB2HSV);
    std::vector<cv::Mat> hsv_split;
    cv::split(hsv_image, hsv_split);
    cv::Scalar sum_saturation = cv::sum(hsv_split[2]);  // Sum the brightness values
    int area = this->shape * this->shape;
    float avg_saturation = sum_saturation[0] / area;

    int sat_low = (int)(avg_saturation * 1.3);
    int val_low = 140;

    cv::Mat green_mask, yellow_mask, red_mask, off_mask;
    // Green
    cv::Scalar lower_green = cv::Scalar(low_green, sat_low, val_low);
    cv::Scalar upper_green = cv::Scalar(up_green, 255, 255);
    inRange(hsv_image, lower_green, upper_green, green_mask);
    int sum_green = countNonZero(green_mask);

    // Yellow
    cv::Scalar lower_yellow = cv::Scalar(low_yellow, sat_low, val_low);
    cv::Scalar upper_yellow = cv::Scalar(up_yellow, 255, 255);
    inRange(hsv_image, lower_yellow, upper_yellow, yellow_mask);
    int sum_yellow = countNonZero(yellow_mask);

    // Red
    cv::Scalar lower_red = cv::Scalar(low_red, sat_low, val_low);
    cv::Scalar upper_red = cv::Scalar(up_red, 255, 255);
    inRange(hsv_image, lower_red, upper_red,red_mask);
    int sum_red = countNonZero(red_mask);

    // Off
    cv::Scalar lower_off = cv::Scalar(low_off, sat_low, val_low);
    cv::Scalar upper_off = cv::Scalar(up_off, 255, 46);
    inRange(hsv_image, lower_off, upper_off, off_mask);
    int sum_off = countNonZero(off_mask);

    M[0]=sum_red;
    M[1]=sum_yellow;
    M[2]=sum_green;
    M[3]=sum_off;
    return M;
    
}

int ** TrafficLightsClassifier::combine_circles(vector<cv::Vec3f> circles,cv::Mat image){
    int** result=new int*[circles.size()];
    for(int i=0;i<circles.size();i++){
        result[i]=new int[4];
    }
    
    int y_min,y_max,x_min,x_max;
    for(size_t i=0;i<circles.size();i++){
        y_min=(int(circles[i][1])-int(circles[i][2]))>=0 ? int(circles[i][1])-int(circles[i][2]) : 0;
        y_max=(int(circles[i][1])+int(circles[i][2]))<=image.size[0] ? int(circles[i][1])+int(circles[i][2]) : image.size[0];
        x_min=(int(circles[i][0])-int(circles[i][2]))>=0 ? int(circles[i][0])-int(circles[i][2]) : 0;
        x_max=(int(circles[i][0])+int(circles[i][2]))<=image.size[1] ? int(circles[i][0])+int(circles[i][2]) : image.size[1];
        cv::Rect rect(x_min,y_min,x_max-x_min,y_max-y_min);
        cv::Mat image_roi = image(rect);   //rect既是要截取的区域
        int* res=red_green_yellow(image_roi);
        for(int j=0;j<4;j++){
            result[i][j]=res[j];
        }
    }
    return result;
}

int TrafficLightsClassifier::estimate_label(int *result[4]){
    int sum_red=0;
    int sum_yellow=0;
    int sum_green=0;
    int sum_off=0;

    for (int i=0;i<sizeof(result[0]) / sizeof(result[0][0]);i++){
        sum_red+=result[i][0];
        sum_yellow+=result[i][1];
        sum_green+=result[i][2];
        sum_off+=result[i][3];
    }

    vector<int> label_values {sum_off,sum_red,sum_green,sum_yellow};
    int maxPosition = max_element(label_values.begin(),label_values.end()) - label_values.begin();

    return maxPosition-1;
}

vector<cv::Vec3f> TrafficLightsClassifier::hough_circles(cv::Mat gray){
    vector<cv::Vec3f> circles;
    HoughCircles(gray, circles,cv::HOUGH_GRADIENT, 1,20,100,20,30,60);
    
    if(circles.empty()){
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(10);
        clahe->setTilesGridSize(cv::Size(3,3));
        cv::Mat imgEquA;
        clahe->apply(gray, imgEquA);
        HoughCircles(gray, circles,cv::HOUGH_GRADIENT, 1,20,100,5,20,50);
    }
    
    return circles;
}

vector<TrafficLightsParams> TrafficLightsClassifier::traffic_lights_result(cv::Mat image,const vector<float> traffic_lights_locations){
    cv::Mat res_img,gray,rgb_image_roi;
    rgb_image_roi = image(cv::Rect(traffic_lights_locations[0], traffic_lights_locations[1], traffic_lights_locations[2], \
        traffic_lights_locations[3]));
    resize(rgb_image_roi,res_img,cv::Size(256,256));
    cvtColor(res_img, gray, cv::COLOR_RGBA2GRAY);

    
    vector<cv::Vec3f> circles;
    circles=hough_circles(gray);
    int **result=combine_circles(circles, res_img);
    int label_value=estimate_label(result);
    TRAFFIC_LIGHTS_TYPE label=TRAFFIC_LIGHTS_TYPE(label_value);
    int target_id =0;
    
    vector<TrafficLightsParams> res={{target_id,label,traffic_lights_locations}};
    return res;
}

