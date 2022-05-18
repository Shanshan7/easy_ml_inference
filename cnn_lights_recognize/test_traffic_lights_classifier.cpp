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
#include <cstring>



int main()
{
    
//    string img_path="/Users/zhangzikai/Downloads/traffic_light_dataset/JPEGImages/000red/traffic_light_0001.jpg";
    string video_path = "/home/ai/zzk/1.mp4";
    string txt_path="/home/ai/zzk/easy_ml_inference/cnn_lights_recognize/1.txt";
    
    TrafficLightsClassifier traffic_lights_classifier;

    cv::VideoCapture cap(video_path);
    cv::Mat frame;
    
    std::ifstream ReadFile;
    ReadFile.open(txt_path,ios::in);
    
    vector<vector<float>> boxes;
    std::string out;
    

    if(ReadFile.fail())
        {
            return 0;
        }
    else{
        while(!ReadFile.fail()){
            getline(ReadFile, out, '\n');
            // std::cout<<out<<std::endl;
    
            std::istringstream ss(out);
        
            vector<float> nums;
            std::string num;

            while(ss >> num) {
                nums.push_back(stoi(num));
            }
//            std::cout<<nums[0]<<std::endl;
            boxes.push_back(nums);
            // break;
        
        }
        std::cout<<boxes.size()<<std::endl;
    }
//
    ReadFile.close();

    

    int frame_num=cap.get(cv::CAP_PROP_FRAME_COUNT);

    for (int i=0;i<frame_num-1;i++){
        cap.read(frame);
//        cv::Rect rect(boxes[i][0],boxes[i][1],boxes[i][0]+boxes[i][2],boxes[i][1]+boxes[i][3]);
//        cv::Mat frame_roi = frame(rect);

        // for(int j=0;j<boxes[i].size();j++){
        //     std::cout<<boxes[i][j]<<' ';
        // }
        // std::cout<<std::endl;


        vector<TrafficLightsParams> result=traffic_lights_classifier.traffic_lights_result(frame, boxes[i],false);
        
        std::string text = to_string(result[0].traffic_lights_type);

        std::cout<<text<<std::endl;

        std::cout<<i<<std::endl;


        // std::cout<<result[0].traffic_lights_location[0][0]<<endl;
        // std::cout<<result[0].traffic_lights_location[0][1]<<endl;
        // std::cout<<result[0].traffic_lights_location[0][2]<<endl;
        // std::cout<<result[0].traffic_lights_location[0][3]<<endl;
        

        

        // cv::putText(frame,text, cv::Point(8, 40), cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

        // cv::imshow("frame", frame);
        // cv::waitKey(0);

        

        i++;   
    }


    std::cout<<"finished"<<std::endl;

    // cap.release();
    return 0;
    
    
    
}
