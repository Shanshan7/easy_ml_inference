#pragma once

#include <ros/ros.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include "kalman.h"
#define ElemType int

using namespace cv;
using namespace std;

/**
 * @brief 卡尔曼预测
 *
 * @param PointList 真实坐标点list
 * @param point_new 插入新真实点
 * @param point_num 真实坐标数
 * @param predict_point_num 预测坐标数
 * @return std::vector<cv::Point> 预测点list
 */
std::vector<cv::Point2f> PredictionPiont(std::vector<cv::Point2f> PointList, Point2f point_new, int point_num, int predict_point_num)
{
    int iKalmanMode = 2;
    bool bFullDebugOn = false;
    
    kalmantracking::kalman oKF(iKalmanMode, bFullDebugOn);


    PointList.insert(PointList.begin() + point_num, point_new);

    PointList.erase(PointList.begin());
    std::vector<cv::Point2f> AllPointList = PointList;
    for (int i=0;i<point_num;i++)
    {
        Point2f p;
        p.x=0;
        p.y=0;
        AllPointList.push_back(p);
    }

    
    for (int i = 0; i < (point_num + predict_point_num); i++)
    // for (int i = 0; i < (point_num + predict_point_num); i++)
    {
        
     
        Point2f ball = AllPointList[i];
        oKF.Predict(ball); //卡尔曼预测，当点为（0，0）时，根据之前坐标预测


        
        
    }
    
    return oKF.getPredictedList();
    // return oKF.getCorrectedList();
}
