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

#include <dirent.h>

//Boost
#include <boost/tokenizer.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include "lonlat2utm.h"
#include "readparam.h"
#include "tracker.h"
#include "smoke.h"


class PerceptionCameraProcess
{
  public:
    // PerceptionCameraProcess() = default;
    // ~PerceptionCameraProcess() = default;
    PerceptionCameraProcess();
    ~PerceptionCameraProcess();

    /**
     * @brief init probabilistic method, init HMAssociation, PbfGatekeeper
     * DstTypeFusion, DstExistanceFusion and PbfTracker
     * @param no paramenters
     */
    void init();
    /**
     * @brief Fusion interface function
     * @param sensor_frame fusion input sensor data
     * @param fused_objects output fused object list
     */
    bool process(cv::Mat& image);
    void getObjects(std::vector<DetectStruct> &camera_objects_);

  private:
    void draw3dbox(DetectStruct &det, cv::Mat& image, std::vector<int>& color, int id);
    void transformTo2dbox();
    void load_offline_files();

    Tracker *tracker;
  #ifdef USE_SMOKE
    SMOKE *detector;
  #endif
  
    cv::Mat intrinsic_;
    Eigen::Matrix<double, 3, 4> camera_intrinsic_;
    Eigen::Matrix4d rt_cam_to_lidar_;
    Eigen::Matrix4d rt_lidar_to_cam_;
    Eigen::Matrix4d rt_imu_to_velo_params_;

    std::unordered_map<int, std::vector<DetectStruct>> input_dets;
    std::vector<Eigen::VectorXd> result;
    std::vector<std::string> gps_data;
    std::unordered_map<std::string, int> classname2id;
    int frame;
    std::string root_path;
    float time;
    boost::char_separator<char> sep { " " };
    std::unordered_map<int, std::vector<int>> idcolor;

    Eigen::Isometry3d porigion;
    Eigen::Isometry3d translate2origion;
    Eigen::Isometry3d origion2translate;
};