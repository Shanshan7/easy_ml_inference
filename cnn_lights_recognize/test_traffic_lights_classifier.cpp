#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "traffic_lights_classifier.h"
#include <fstream>
#include <string>
#include <sstream>



int main()
{
    
    string img_path="/Users/zhangzikai/Downloads/traffic_light_dataset/JPEGImages/000red/traffic_light_0001.jpg";
    string video_path = "/Users/zhangzikai/Desktop/traffic_light_mp4/1.mp4";
    string txt_path="/Users/zhangzikai/Desktop/easy_ml_inference/cnn_lights_recognize/1.txt";
    
    TrafficLightsClassifier traffic_lights_classifier;
    
    cv::VideoCapture cap(video_path);
    cv::Mat frame;
    
    std::ifstream ReadFile;
    ReadFile.open(txt_path,ios::in);
    
    vector<vector<int>> boxes;
    std::string out;
    

    if(ReadFile.fail())
        {
            return 0;
        }
    else{
        while(!ReadFile.fail()){
            getline(ReadFile, out, '\n');
            std::cout<<out<<std::endl;
    
            std::istringstream ss(out);
        
            vector<int> nums;
            std::string num;

            while(ss >> num) {
                nums.push_back(stoi(num));
            }
            std::cout<<nums[0]<<std::endl;
            break;
        }
    }
//
    ReadFile.close();

    int i=0;
        
    while(true)
    {
        cap.read(frame);
        cv::Rect rect(boxes[i][0],boxes[i][1],boxes[i][0]+boxes[i][2],boxes[i][1]+boxes[i][3]);
        cv::Mat frame_roi = frame(rect);

        vector<TrafficLightsParams> result=traffic_lights_classifier.traffic_lights_result(frame_roi, traffic_lights_classifier.traffic_lights_locations);

        std::string text = to_string(result[0].traffic_lights_type);

        cv::putText(frame_roi,text, cv::Point(8, 40), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

        cv::imshow("frame_roi", frame_roi);
        cv::waitKey(0);

        i++;
    }
    
    cap.release();
    return 0;
    
    
    
}
