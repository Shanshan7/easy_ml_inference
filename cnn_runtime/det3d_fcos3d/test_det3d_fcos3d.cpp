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

#include <sys/time.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


int main()
{
    FCOS3D *detector;
    detector = new FCOS3D("/docker_data/easy_ml_inference/cnn_runtime/det3d_fcos3d/fcos3d_1.6_100.trt8");

    // single image inference
    // std::string impath = "/docker_data/data/front_2M/front_2M_02/18.png";
    // std::cout << "img path: " << impath << std::endl;
    // cv::Mat rgbimage = cv::imread(impath);

    // detector->detect(rgbimage);
    // std::cout << "object counts: " << detector->result_bboxes.size() << std::endl;
    // detector->ShowResult(rgbimage);

    struct timeval tm1, tm2;

    for(int i = 1; i < 195; i++)
    {
        std::stringstream temp_str;
        temp_str.str("");
        temp_str << i;
        std::string imag_name = temp_str.str();

        std::string impath = "/docker_data/data/front_2M/front_2M_02/" + imag_name + ".png";
        std::cout << "img path: " << impath << std::endl;
        cv::Mat rgbimage = cv::imread(impath);

        gettimeofday(&tm1, 0);
        int64_t re_start = (((int64_t) tm1.tv_sec) * 1000 * 1000 + tm1.tv_usec);
        detector->detect(rgbimage);
        gettimeofday(&tm2, 0);
        int64_t re_stop = (((int64_t) tm2.tv_sec) * 1000 * 1000 + tm2.tv_usec);
        std::cout << "object counts: " << detector->result_bboxes.size() << std::endl;
        std::cout << "[Time cost]: " << (re_stop - re_start) / 1000 << " ms" << std::endl;
        detector->ShowResult(rgbimage);
    }

    return 0;
}