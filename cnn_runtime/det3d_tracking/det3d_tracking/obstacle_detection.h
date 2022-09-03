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
#include "smoke.h"
#include "tracker.h"


struct pose{
	double x;
	double y;
	double heading;
};

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

  private:
    void draw3dbox(DetectStruct &det, cv::Mat& image, std::vector<int>& color, int id);
    void load_offline_files();

    Tracker *tracker;
    SMOKE *detector;

    // std::mutex data_mutex_;
    // std::mutex fuse_mutex_;
    // bool started_ = false;
    // TrafficLightScenePtr scenes_;
    // std::vector< std::shared_ptr< TrafficLightTrack > > trackers_;
    // std::unique_ptr< HMTrackersObjectsAssociation > matcher_;
    // FusionParams params_;

    cv::Mat intrinsic;
    std::unordered_map<int, std::vector<DetectStruct>> Inputdets;
    std::vector<std::string> gpsdata;
    std::unordered_map<std::string, int> classname2id;
    int frame;
    std::string root_path;
    float time;
    boost::char_separator<char> sep { " " };

    Eigen::Isometry3d porigion;
    // Eigen::Isometry3d translate2origion;
    std::unordered_map<int, std::vector<int>> idcolor;
};