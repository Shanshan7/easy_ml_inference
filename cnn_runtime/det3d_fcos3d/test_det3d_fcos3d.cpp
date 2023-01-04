/******************************************************************************
**                                                                           **
** Copyright (C) Joyson Electronics (2022)                                   **
**                                                                           **
** All rights reserved.                                                      **
**                                                                           **
** This document contains proprietary information belonging to Joyson        **
** Electronics. Passing on and copying of this document, and communication   **
** of its contents is not permitted without prior written authorization.     **
**                                                                           **
******************************************************************************/

#include "fcos3d.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


int main()
{
    FCOS3D *detector;
    detector = new FCOS3D("/docker_data/easy_ml_inference/cnn_runtime/det3d_fcos3d/fcos3d.trt8");

    // single image inference
    std::string impath = "/docker_data/data/front_2M_nuscenes/samples/CAM_FRONT/n015-2018-10-02-10-50-40+0800__CAM_FRONT__1538448764012460.jpg";
    std::cout << "img path: " << impath << std::endl;
    cv::Mat rgbimage = cv::imread(impath);

    detector->detect(rgbimage);
    std::cout << "object counts: " << detector->result_bboxes.size() << std::endl;
    detector->ShowResult(rgbimage);

    // for(int i = 1; i < 195; i++)
    // {
    //     std::stringstream temp_str;
    //     temp_str.str("");
    //     temp_str << i;
    //     std::string imag_name = temp_str.str();

    //     std::string impath = "/docker_data/data/front_2M/front_2M_03/" + imag_name + ".png";
    //     std::cout << "img path: " << impath << std::endl;
    //     cv::Mat rgbimage = cv::imread(impath);

    //     detector->detect(rgbimage);
    //     std::cout << "object counts: " << detector->result_bboxes.size() << std::endl;
    //     detector->ShowResult(rgbimage);
    // }

    return 0;
}