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

#include "obstacle_detection.h"


int main()
{
  int frame = 0;
  std::string image_file_path = "/data/kitti_mini0020/image_02/";
  // PerceptionCameraProcess *perception_camera_process = nullptr;
  PerceptionCameraProcess perception_camera_process;

  // perception_camera_process->init();
  perception_camera_process.init();
  std::cout << "Init done" << std::endl;

  while (true)
  {
	std::stringstream temp_str;
    temp_str.str("");
    int index = 1000000 + frame;
    temp_str << index;
    std::string image_name_long = temp_str.str();
    std::string imag_name = image_name_long.substr(1, 6);

    std::string impath = image_file_path + imag_name + ".png";
    std::cout << "impath: " << impath << std::endl;
    cv::Mat rgbimage = cv::imread(impath);
    perception_camera_process.process(rgbimage);

    frame++;
  }

  return 0;
}