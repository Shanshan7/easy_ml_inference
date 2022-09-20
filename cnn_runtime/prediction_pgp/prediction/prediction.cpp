#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
// #include "perception_msgs/FuseObjectList.h"
#include <perception_msgs/FuseObjectList.h>
#include <perception_msgs/PerceptionObjectList.h>
#include <perception_msgs/PredictedObjectList.h>
#include <tf/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cmath>
#include <map>

#include "lib/predictor/kalman_predictor/predictpoint.h"


std::map<int16_t, std::vector<cv::Point2f>> pointlist; //全局变量
void Pointlist(std::map<int16_t, std::vector<cv::Point2f>>& pointlist) {
    for (int i = 0; i < 500; i++) {
        std::vector<cv::Point2f> pointlist[i]; //障碍物id:0-500
    }
}

class Prediction {
private:
    ros::NodeHandle n;
    ros::Subscriber data_info_sub;
    ros::Publisher msg_pub;
    perception_msgs::PredictedObjectList precicted_msg;
    std::int8_t point_num = 30;
    std::int8_t prediction_point_num = 25;
    // perception2fusion
    ros::Publisher fusionPub;
    ros::Subscriber matrix_objectlist;
    ros::Publisher Vus_msg_pub;
    ros::Publisher Vus_msg_pub1;

    // 传进来时间间隔
    float time_t_0;

public:
    Prediction() {
        data_info_sub = n.subscribe("/perception_objection_list", 10, &Prediction::dataInfoCallback, this);
        msg_pub = n.advertise<perception_msgs::PredictedObjectList>("prediction_msg", 1000);
        Vus_msg_pub1 = n.advertise<visualization_msgs::MarkerArray>("Veus_prediction_msg_kalman", 10);
    }
    void dataInfoCallback(const perception_msgs::PerceptionObjectList::ConstPtr& msg1);
};

// msg转换
void msgtransform(const perception_msgs::PerceptionObjectList::ConstPtr& msg, perception_msgs::FuseObjectList& fuseojectlist) {
    fuseojectlist.fuse_obj_num = msg->perceptionobjects.size();
    fuseojectlist.header = msg->timestamp;
    // std::cout<<"fuseojectlist.fuse_obj_num:  "<<fuseojectlist.fuse_obj_num<<std::endl;
    for (int i = 0; i < msg->perceptionobjects.size(); i++) {
        perception_msgs::FuseObject fuse_object;
        // 1.探测到该目标的传感器
        fuse_object.detect_sensor_current = 1;
        // 2.在目标的生命周期内，曾经探测到该目标的传感器
        fuse_object.detect_sensor_history = 1;
        // 3. 时间戳
        // fuse_object.object_time_stamp = msg->timestamp.stamp;
        // 4. 目标类别
        fuse_object.object_type = msg->perceptionobjects[i].obj_type;

        // 5. 目标存活时间
        fuse_object.object_age = msg->perceptionobjects[i].object_age;
        // 6. ID号
        fuse_object.object_id = msg->perceptionobjects[i].obj_id;

        // 3D障碍物边框信息
        fuse_object.object_rect3d = msg->perceptionobjects[i].rect3d;
        std::cout << msg->perceptionobjects[i] << std::endl;
        //这里做了一些变化，把position的坐标传给世界坐标系
        fuse_object.object_rect3d_world = msg->perceptionobjects[i].rect3d;
        // std::cout<<perceptionobjects[i].rect3d<<std::endl;
        fuse_object.object_rect3d_world.center.x = msg->perceptionobjects[i].wheel_position.x;

        fuse_object.object_rect3d_world.center.y = msg->perceptionobjects[i].wheel_position.y;
        // fuse_object.object_rect3d_world.center.x=(msg->perceptionobjects[i].box2d.min_point.x+msg->perceptionobjects[i].box2d.max_point.x)/2;
        // fuse_object.object_rect3d_world.center.y=(msg->perceptionobjects[i].box2d.min_point.y+msg->perceptionobjects[i].box2d.max_point.y)/2;
        // std::cout<<fuse_object.object_rect3d_world<<std::endl;
        //目标类别置信度
        //
        fuse_object.object_obj_confidence = msg->perceptionobjects[i].confidence;
        //速度
        fuse_object.object_velocity = msg->perceptionobjects[i].velocity;
        //状态
        if (fuse_object.object_velocity.x == 0 && fuse_object.object_velocity.y == 0) {
            fuse_object.object_motion_status = 2;
        } else {
            fuse_object.object_motion_status = 1;
        }
        // std::cout << msg->perceptionobjects[i].wheel_position.x  << std::endl;
        fuseojectlist.FuseObjects.push_back(fuse_object);
    }
}

void Prediction::dataInfoCallback(const perception_msgs::PerceptionObjectList::ConstPtr& msg1) {
    /*
    为发布的precicted_msg赋值const perception_msgs::FuseObjectList::ConstPtr &msg
    */
    perception_msgs::FuseObjectList msg;

    msgtransform(msg1, msg);
    int fuse_obj_num = msg.fuse_obj_num;
    std::vector<perception_msgs::FuseObject> FuseObjects = msg.FuseObjects;
    precicted_msg.obj_num = fuse_obj_num;
    // std::cout<<"fuse_obj_num:  "<<fuse_obj_num<<std::endl;
    precicted_msg.predicted_objects.resize(fuse_obj_num);

    visualization_msgs::MarkerArray marker_array_kalman;
    tf2::Quaternion myQuaternion;

    for (int i = 0; i < fuse_obj_num; i++) {
        perception_msgs::PredictedTrajectory predicted_traj_;

        precicted_msg.predicted_objects[i].obj_id = msg.FuseObjects[i].object_id;

        precicted_msg.predicted_objects[i].pred_obj_time_stamp = (uint64_t)0;
        precicted_msg.predicted_objects[i].traj_num = (uint8_t)1;

        predicted_traj_.motion_intent = (int8_t)0;
        predicted_traj_.confidence = 100;
        predicted_traj_.traj_start_time = msg.FuseObjects[i].object_time_stamp;
        ;
        predicted_traj_.traj_end_time = msg.FuseObjects[i].object_time_stamp + (time_t_0 * 25);
        ;
        predicted_traj_.traj_point_num = 25;
        precicted_msg.predicted_objects[i].predicted_trajs.push_back(predicted_traj_);
        float pointx = float(FuseObjects[i].object_rect3d_world.center.x);

        float pointy = float(FuseObjects[i].object_rect3d_world.center.y);
        cv::Point2f ball(pointx, pointy);

        /*
            如果真实值不足point_num，添加真实值
            否则调用卡尔曼滤波进行预测prediction_point_num个预测点
    */
        if (pointlist[FuseObjects[i].object_id].size() < point_num) {
            pointlist[FuseObjects[i].object_id].push_back(ball);

        }

        else {
            std::vector<cv::Point2f> prediction = PredictionPiont(pointlist[FuseObjects[i].object_id], ball, point_num, prediction_point_num);

            visualization_msgs::Marker mobjectorig;
            mobjectorig.header.stamp = ros::Time::now();
            mobjectorig.header.frame_id = "rslidar";
            mobjectorig.ns = "objects";
            mobjectorig.id = i + 900;
            mobjectorig.type = 1;   // Cube
            mobjectorig.action = 0; // add/modify
            mobjectorig.scale.x = 0.5;
            mobjectorig.scale.y = 0.5;
            mobjectorig.scale.z = 0.1;
            myQuaternion.setRPY(0, 0, FuseObjects[i].object_rect3d_world.heading);

            mobjectorig.pose.orientation.w = myQuaternion.getW();
            mobjectorig.pose.orientation.x = myQuaternion.getX();
            mobjectorig.pose.orientation.y = myQuaternion.getY();
            mobjectorig.pose.orientation.z = myQuaternion.getZ();
            mobjectorig.lifetime = ros::Duration(0.2);
            mobjectorig.frame_locked = false;

            mobjectorig.pose.position.x = ball.x;
            mobjectorig.pose.position.y = ball.y;
            mobjectorig.pose.position.z = 0;
            mobjectorig.color.r = 1;
            mobjectorig.color.g = 0;
            mobjectorig.color.b = 0;
            mobjectorig.color.a = 0.6;
            visualization_msgs::Marker mtext;

            mtext.header.frame_id = "rslidar";
            mtext.ns = "rslidar";
            // set marker type
            mtext.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            mtext.pose.orientation.w = 1.0; //文字的方向
            mtext.id = i + 100; //用来标记同一帧不同的对象，如果后面的帧的对象少于前面帧的对象，那么少的id将在rviz中残留，所以需要后续的实时更新程序
            mtext.scale.x = 1.5;
            mtext.scale.y = 1.5;
            mtext.scale.z = 0.8; //文字的大小
            mtext.lifetime = ros::Duration(0.2);
            mtext.frame_locked = false;
            mtext.color.b = 25;
            mtext.color.g = 25;
            mtext.color.r = 240; //文字的颜色

            mtext.color.a = 1;
            geometry_msgs::Pose pose;
            pose.position.x = ball.x;
            pose.position.y = ball.y;
            pose.position.z = 0;
            mtext.text = "id:" + std::to_string(FuseObjects[i].object_id) + " conf:" + std::to_string(FuseObjects[i].object_obj_confidence); //文字内容
            mtext.pose = pose;                                                                                                               //文字的位置
            // std::cout<<prediction.size()<<std::endl;
            for (int j = 0; j < prediction_point_num; j++) {
                perception_msgs::PredictedTrajectoryPoint point2;
                point2.x = prediction[j].x;
                point2.y = prediction[j].y;
                precicted_msg.predicted_objects[i].predicted_trajs[0].predicted_traj_points.push_back(point2);
                //
                precicted_msg.predicted_objects[i].predicted_trajs[0].predicted_traj_points[j].x = prediction[j].x;
                precicted_msg.predicted_objects[i].predicted_trajs[0].predicted_traj_points[j].y = prediction[j].y;
                float x0;
                float y0;
                if (j < 1) {
                    float x0 = 0;
                    float y0 = 0;
                } else {
                    float x0 = precicted_msg.predicted_objects[i].predicted_trajs[0].predicted_traj_points[j - 1].x;
                    float y0 = precicted_msg.predicted_objects[i].predicted_trajs[0].predicted_traj_points[j - 1].y;
                }

                float x = precicted_msg.predicted_objects[i].predicted_trajs[0].predicted_traj_points[j].x;
                float y = precicted_msg.predicted_objects[i].predicted_trajs[0].predicted_traj_points[j].y;
                precicted_msg.predicted_objects[i].predicted_trajs[0].predicted_traj_points[j].heading = atan2(x - x0, y - y0) / 3.14 * 180;
                precicted_msg.predicted_objects[i].predicted_trajs[0].predicted_traj_points[j].speed = msg.FuseObjects[i].object_velocity_world.x;
                precicted_msg.predicted_objects[i].predicted_trajs[0].predicted_traj_points[j].acceleration = msg.FuseObjects[i].object_acc_world.x;
                precicted_msg.predicted_objects[i].predicted_trajs[0].predicted_traj_points[j].lane_id = msg.FuseObjects[i].lane_id;

                visualization_msgs::Marker mobject2;
                mobject2.header.stamp = ros::Time::now();
                mobject2.header.frame_id = "rslidar";
                mobject2.ns = "objects";
                mobject2.id = 25 * i + j;
                mobject2.type = 1;   // Cube
                mobject2.action = 0; // add/modify

                myQuaternion.setRPY(0, 0, FuseObjects[i].object_rect3d_world.heading);

                mobject2.pose.orientation.w = myQuaternion.getW();
                mobject2.pose.orientation.x = myQuaternion.getX();
                mobject2.pose.orientation.y = myQuaternion.getY();
                mobject2.pose.orientation.z = myQuaternion.getZ();
                mobject2.scale.x = 0.3;
                mobject2.scale.y = 0.3;
                mobject2.scale.z = 0.1;

                mobject2.lifetime = ros::Duration(0.2);
                mobject2.frame_locked = false;

                mobject2.pose.position.x = point2.x;
                mobject2.pose.position.y = point2.y;
                mobject2.pose.position.z = 0;
                mobject2.color.r = 0;
                mobject2.color.g = 1;
                mobject2.color.b = 0;
                mobject2.color.a = 0.7;

                //原始位置
                ROS_INFO("Vusion_Publish prediction Info: obj_id:%d point_idx:%d  point_x:%f point_y:%f", mobject2.id, j, mobject2.pose.position.x,
                         mobject2.pose.position.y);
                marker_array_kalman.markers.push_back(mobject2);
            }
            marker_array_kalman.markers.push_back(mobjectorig);
            marker_array_kalman.markers.push_back(mtext);
        }
    }

    msg_pub.publish(precicted_msg);
    Vus_msg_pub1.publish(marker_array_kalman);
    // r.sleep();
}
int main(int argc, char** argv) {
    Pointlist(pointlist);

    ros::init(argc, argv, "prediction");

    Prediction prediction;

    ros::spin();
    return 0;
}