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
#pragma once

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>


struct DetectStruct{
	int id;
	int classname = 0;
	float score;//detection score
	float z;
	float yaw;
	Eigen::VectorXd position; // x, y
	std::vector<float> box; // 3D box in lidar h、w、l
	std::vector<float> box2D; // 2D box in camera left top point and right down point
	cv::RotatedRect rotbox;
};

struct Box{
	float yaw;
	float length;
	float width;
	float height;
	float z;
};

struct TrackState{
	int Confirmed = 1;
	int UnConfirmed = 2;
	int Delete = 3;
};

struct BboxDim {
    float x;
    float y;
    float z;
};