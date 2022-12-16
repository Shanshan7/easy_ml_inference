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
    detector = new FCOS3D("/docker_data/easy_ml_inference/cnn_runtime/det3d_fcos3d/fcos3d.trt8");

    std::string impath = "/docker_data/data/front_2M/001683.png";
    cv::Mat rgbimage = cv::imread(impath);
    detector->detect(rgbimage);
    detector->ShowResult(rgbimage);

    std::cout << "result num: " << detector->result_bboxes.size() << std::endl;

    // scores
    for(int i = 0; i < detector->result_bboxes.size(); i++)
    {
        std::cout << "detector: "
                  << detector->result_bboxes[i].x << " "
                  << detector->result_bboxes[i].y << " "
                  << detector->result_bboxes[i].depth << " "
                  << detector->result_bboxes[i].w << " "
                  << detector->result_bboxes[i].h << " "
                  << detector->result_bboxes[i].l << " "
                  << detector->result_bboxes[i].rotation << " "
                  << detector->result_bboxes[i].velocity_x << " "
                  << detector->result_bboxes[i].velocity_y << " "
                  << std::endl;
        std::cout << "score: " << detector->result_scores[i] << std::endl;
        std::cout << "direction score: " << detector->result_dir_scores[i] << std::endl;
        std::cout << "label: " << detector->result_labels[i] << std::endl;
    }

    return 0;
}