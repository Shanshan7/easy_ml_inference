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
#include "JoysonPredictionPost.h"

JoysonPredictionPost::JoysonPredictionPost() {}

JoysonPredictionPost::~JoysonPredictionPost() {}

void JoysonPredictionPost::PredictPoints(prediction::PredictedTrajectoryPoint &predicted_point, int i,
                                         float traj_point_num, float x0, float y0, float x, float y, float speed0,
                                         int8_t lane_id, float time0) {
    // predicted_point.x = x;
    // predicted_point.y = y;
    predicted_point.heading = atan2(predicted_point.x - x0, predicted_point.y - y0) / 3.14 * 180;
    predicted_point.speed = (sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0))) / (6 / traj_point_num);
    predicted_point.acceleration = (predicted_point.speed - speed0) / (6 / traj_point_num);
    predicted_point.lane_id = lane_id;
    float point_time = (6 / traj_point_num) * i + time0;
    predicted_point.point_time = point_time;

    // return predicted_point;
}

void JoysonPredictionPost::PredictedTrajectorys(prediction::PredictedTrajectory &Trajectory, uint64_t obj_time_stamp) {
    enum MotionIntent {
        MOTION_INTENT_UNKNOWN = -1,
        MOTION_INTENT_GO_STRAIGHT = 0,
        MOTION_INTENT_TURN_RIGHT = 1,
        MOTION_INTENT_TURN_LEFT = 3,
        MOTION_INTENT_TURN_AROUND = 4
    };

    Trajectory.traj_start_time = obj_time_stamp;
    Trajectory.traj_end_time = obj_time_stamp + 6; // 6s
    Trajectory.confidence = Trajectory.confidence;
    Trajectory.traj_point_num = 12; //预测12 个点

    for (int i = 1; i < Trajectory.traj_point_num; i++) {
        float x0 = Trajectory.predicted_traj_points[i - 1].x;
        float y0 = Trajectory.predicted_traj_points[i - 1].y;
        float speed0 = Trajectory.predicted_traj_points[i - 1].speed;

        prediction::PredictedTrajectoryPoint predicted_point;
        PredictPoints(predicted_point, i, Trajectory.traj_point_num, x0, y0, Trajectory.predicted_traj_points[i].x,
                      Trajectory.predicted_traj_points[i].y, speed0, Trajectory.predicted_traj_points[i].lane_id,
                      obj_time_stamp);
        Trajectory.predicted_traj_points[i] = predicted_point;
    }

    double trajectory = Trajectory.predicted_traj_points[11].heading -
                        Trajectory.predicted_traj_points[0].heading; //最后一个的朝向与第一个点的朝向差值

    if (-30 <= trajectory <= 30) //角度相差小于30，判断为直行
    {
        Trajectory.motion_intent = MOTION_INTENT_GO_STRAIGHT;
    } else if (30 <= trajectory <= 90) {
        Trajectory.motion_intent = MOTION_INTENT_TURN_RIGHT;
    } else if (-90 <= trajectory <= -30) {
        Trajectory.motion_intent = MOTION_INTENT_TURN_LEFT;
    } else if (trajectory >= 90) {
        Trajectory.motion_intent = MOTION_INTENT_TURN_AROUND;
    } else {
        Trajectory.motion_intent = MOTION_INTENT_UNKNOWN;
    }
    // return Trajectory;
}

/**
 * @brief  后处理传值
 *
 * @param predictoutdata PredictedOutData 结构体输入
 * @return prediction::PredictedObject
 */
int JoysonPredictionPost::PredictedPostprocess(prediction::PredictedObject &cluster_object_trajectory) {
    time_t times;
    times = time(NULL);
    prediction::PredictedTrajectory Trajectory;
    cluster_object_trajectory.predicted_obj_time_stamp = times;

    for (int i = 0; i < cluster_object_trajectory.traj_num; i++) {
        PredictedTrajectorys(cluster_object_trajectory.predicted_trajs[i],
                             cluster_object_trajectory.predicted_obj_time_stamp);
    }

    return 0;
}
void JoysonPredictionPost::StructTomsg(prediction::PredictedObject &objectstruct,
                                       perception_msgs::PredictedObject &objectmsg) {
    objectmsg.obj_id = objectstruct.obj_id;
    objectmsg.pred_obj_time_stamp = objectmsg.pred_obj_time_stamp;
    objectmsg.traj_num = objectmsg.traj_num;
    // objectmsg.predicted_trajs=objectstruct.predicted_trajs;

    for (int i = 0; i < objectstruct.traj_num; i++) {
        objectmsg.predicted_trajs[i].confidence = objectstruct.predicted_trajs[i].confidence;
        objectmsg.predicted_trajs[i].motion_intent = objectstruct.predicted_trajs[i].motion_intent;
        objectmsg.predicted_trajs[i].traj_start_time = objectstruct.predicted_trajs[i].traj_start_time;
        objectmsg.predicted_trajs[i].traj_end_time = objectstruct.predicted_trajs[i].traj_end_time;
        objectmsg.predicted_trajs[i].traj_point_num = objectstruct.predicted_trajs[i].traj_point_num;
        for (int k = 0; k < objectstruct.predicted_trajs[i].traj_point_num; k++) {
            objectmsg.predicted_trajs[i].predicted_traj_points[k].acceleration =
                objectstruct.predicted_trajs[i].predicted_traj_points[k].acceleration;
            objectmsg.predicted_trajs[i].predicted_traj_points[k].heading =
                objectstruct.predicted_trajs[i].predicted_traj_points[k].heading;
            objectmsg.predicted_trajs[i].predicted_traj_points[k].lane_id =
                objectstruct.predicted_trajs[i].predicted_traj_points[k].lane_id;
            objectmsg.predicted_trajs[i].predicted_traj_points[k].point_time =
                objectstruct.predicted_trajs[i].predicted_traj_points[k].point_time;
            objectmsg.predicted_trajs[i].predicted_traj_points[k].speed =
                objectstruct.predicted_trajs[i].predicted_traj_points[k].speed;
            objectmsg.predicted_trajs[i].predicted_traj_points[k].x =
                objectstruct.predicted_trajs[i].predicted_traj_points[k].x;
            objectmsg.predicted_trajs[i].predicted_traj_points[k].y =
                objectstruct.predicted_trajs[i].predicted_traj_points[k].y;
        }
    }
}