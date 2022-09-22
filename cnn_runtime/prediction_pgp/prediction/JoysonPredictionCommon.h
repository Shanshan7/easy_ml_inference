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
#include <string>
#include <fstream>
#include <vector>
#include <time.h>
// #include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/core.hpp>

#define OBJECT_NUM 1

namespace prediction
{
    // ----------------------------prediction object define-------------------------------
    enum MotionIntent
    {
        MOTION_INTENT_UNKNOWN = -1,
        MOTION_INTENT_GO_STRAIGHT = 0,
        MOTION_INTENT_TURN_RIGHT = 1,
        MOTION_INTENT_TURN_LEFT = 3,
        MOTION_INTENT_TURN_AROUND = 4
    };

    struct PredictedTrajectoryPoint
    {
        float x;
        float y;
        float heading;
        float speed;
        float acceleration;
        int lane_id;
        uint64 point_time;
    };

    struct PredictedTrajectory
    {
        int8_t motion_intent;
        uint64_t traj_start_time;
        uint64_t traj_end_time;
        uint8_t confidence;
        uint16_t traj_point_num;
        std::vector<PredictedTrajectoryPoint> predicted_traj_points;
    };

    struct PredictedObject
    {
        int8_t obj_id;
        uint64_t predicted_obj_time_stamp;
        uint8_t traj_num;
        std::vector<PredictedTrajectory> predicted_trajs;
    };

    // struct PredictedObjects
    // {
    //     uint8_t obj_num;
    //     std::vector<PredictedObjectList> predicted_object;
    // };

    // struct TrajectoryPoint
    // {
    //     float x;
    //     float y;
    //     float heading;
    //     float speed;
    //     float acceleration;
    //     int lane_id;
    // };

    // struct PredictedTrajectory
    // {
    //     MotionIntent motion_intent;
    //     uint64_t traj_start_time;
    //     uint64_t traj_end_time;
    //     uint8_t confidence;
    //     uint16_t traj_point_num;
    //     std::vector<TrajectoryPoint> traj_point_list;
    // };

    // struct PredictedObject
    // {
    //     int8_t obj_id;
    //     uint64_t predicted_obj_time_stamp;
    //     uint8_t traj_num;
    //     std::vector<PredictedTrajectory> predicted_traj_list;
    };

    struct PredictedNetParam
    {
        const char *model_path;

        std::vector<const char *> input_node_names;
        std::vector<const char *> output_node_names;

        std::array<int64_t, 3> input_shape_{1, 5, 5};
        std::array<int64_t, 4> input_shape_7{1, 164, 20, 6};
        std::array<int64_t, 4> lane_node_masks_input_shape{1, 164, 20, 6};
        std::array<int64_t, 4> f4_input_shape{1, 84, 5, 5};
        std::array<int64_t, 4> nbr_vehicle_masks_input_shape{1, 84, 5, 5};
        std::array<int64_t, 4> f6_input_shape{1, 77, 5, 5};
        std::array<int64_t, 4> nbr_ped_masks_input_shape{1, 77, 5, 5};
        std::array<int64_t, 3> f8_input_shape{1, 164, 84};
        std::array<int64_t, 3> f9_input_shape{1, 164, 77};
        std::array<int64_t, 3> input_shape_9{1, 164, 15};
        std::array<int64_t, 3> edge_type_input_shape{1, 164, 15};
        std::array<int64_t, 3> init_node_input_shape{1, 164};
    };

// } // namespace prediction
