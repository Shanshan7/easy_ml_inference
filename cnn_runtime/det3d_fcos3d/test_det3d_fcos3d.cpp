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
    cv::Mat intrinsic_ = (cv::Mat_<float>(3, 3) << 1015.2, 0.0, 960.2334, 
                                    0.0, 1015.5, 487.1393,
                                    0.0, 0.0, 1.0);
    FCOS3D *detector;
    detector = new FCOS3D("/docker_data/easy_ml_inference/cnn_runtime/det3d_tracking/data/models/smoke_dla34.trt8", 
	                     intrinsic_);

    std::string impath = "/docker_data/data/front_2M/imag23.png";
    cv::Mat rgbimage = cv::imread(impath);
    detector->detect(rgbimage);

    return 0;
}