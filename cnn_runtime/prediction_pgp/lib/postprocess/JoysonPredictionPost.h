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

// #include <perception_msgs/PredictedObjectList.h>

#include "JoysonPredictionCommon.h"
// #include "JoysonPredictionPre.h"


struct ClusterDataInfo
{
    int cluser_cnts;
    std::vector<float> cluster_lbls;
    std::vector<float> cluster_ranks;
};
/*
     prediction::PredictedObject单个障碍物的信息
*/
class JoysonPredictionPost
{
private:
    /* data */
private:
    /* function */
    void ClusterAndRank(int num_clusters, float &data, ClusterDataInfo cluster_data);

    void PredictPoints(prediction::PredictedTrajectoryPoint &predicted_point,
                       int i, float traj_point_num, float x0, float y0, 
                       float x, float y, float speed0, int8_t lane_id, float time0);
    void PredictedTrajectorys(prediction::PredictedTrajectory &Trajectory,
                              uint64_t obj_time_stamp);

public:
    // prediction::PredictedTrajectoryList Trajectory;
    // prediction::TrajectoryPointList predicted_point;
    // prediction::PredictedObjects objects; 
    // std::vector<prediction::PredictedObjects> &cluster_object_trajectory;

public:
    JoysonPredictionPost();
    ~JoysonPredictionPost();

    /* function */
    // int PGPNetRun(PredictionNetInput &prediction_net_input, float &traj);
    void ClusterTraj(float &traj, std::vector<prediction::PredictedObject> &cluster_object_trajectory);
    int PredictedPostprocess(prediction::PredictedObject &cluster_object_trajectory);
    // void StructTomsg(prediction::PredictedObject &spredictedtruct,perception_msgs::PredictedObject &objectmsg);
};