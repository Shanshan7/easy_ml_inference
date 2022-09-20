// #include <perception_msgs/FuseObjectList.h>
// #include <perception_msgs/PerceptionObjectList.h>
// #include <perception_msgs/PredictedObjectList.h>
// #include <tf/transform_broadcaster.h>
// #include <tf2/LinearMath/Quaternion.h>
// #include <tf2_ros/transform_broadcaster.h>
// #include <visualization_msgs/Marker.h>
// #include <visualization_msgs/MarkerArray.h>
// #include <cmath>
// #include <map>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>
// #include "perception_prediction/predictpoint.hpp"
// #include "ros/ros.h"


// std::map<int16_t, std::vector<perception_msgs::FuseObject>> objectlist; //全局变量
// void Objectlist(std::map<perception_msgs::FuseObject>> &objectlist) {
//     for (int i = 0; i < 500; i++) {
//         std::vector<perception_msgs::FuseObject> objectlist[i]; //障碍物id:0-500
//     }
// }

// class Prediction {
// private:
//     ros::NodeHandle n;
//     ros::Subscriber data_info_sub;
//     ros::Publisher msg_pub;
//     perception_msgs::PredictedObjectList precicted_msg;
//     std::int8_t post_msg_num = 20;
//     std::int8_t point_num = 30;
//     std::int8_t prediction_point_num = 25;

// public:
//     Prediction() {
//         data_info_sub = n.subscribe("/perception_objection_list", 10, &Prediction::dataInfoCallback, this);
//         msg_pub = n.advertise<perception_msgs::PredictedObjectList>("prediction_msg", 1000);
//         Vus_msg_pub1 = n.advertise<visualization_msgs::MarkerArray>("Veus_prediction_msg", 10);
//     }
//     void dataInfoCallback(const perception_msgs::PerceptionObjectList::ConstPtr& msg1);
// };

// void Prediction::dataInfoCallback(const perception_msgs::PerceptionObjectList::ConstPtr& msg1) {
//     /*
//     为发布的precicted_msg赋值const perception_msgs::FuseObjectList::ConstPtr &msg
//     */
//     perception_msgs::FuseObjectList& msg;
//     int fuse_obj_num = msg.fuse_obj_num;
//     std::vector<perception_msgs::FuseObject> FuseObjects = msg.FuseObjects;
//     precicted_msg.obj_num = fuse_obj_num;
//     // std::cout<<"fuse_obj_num:  "<<fuse_obj_num<<std::endl;
//     precicted_msg.predicted_objects.resize(fuse_obj_num);

//     visualization_msgs::MarkerArray marker_array_pgp;
//     tf2::Quaternion myQuaternion;

//     for (int i = 0; i < fuse_obj_num; i++) {
//         if (objectlist[FuseObjects[i].object_id].size() < post_msg_num) {
//             objectlist[FuseObjects[i].object_id].push_back(FuseObjects[i]);
//         } else {
//             objectlist[FuseObjects[i].object_id].push_back(FuseObjects[i]);
//             objectlist[FuseObjects[i].object_id].erase(objectlist[FuseObjects[i].object_id].begin());

//             /******************************************************************
//             precicted_msg.predicted_objects[i] = PgpPrediction(objectlist[FuseObjects[i].object_id]);//将该目标当前与历史信息输入

//             *******************************************************************/

//             for (int a = 0; a < precicted_msg.obj_num; a++) {
//                 for (int b = 0; b < precicted_msg.predicted_objects[a].traj_num; b++) {
//                     for (int c = 0; c < precicted_msg.predicted_objects[a].predicted_trajs[b].traj_point_num; c++) {
//                         if (c = 0) {
//                             visualization_msgs::Marker mobjectorig;
//                             mobjectorig.header.stamp = ros::Time::now();
//                             mobjectorig.header.frame_id = "rslidar";
//                             mobjectorig.ns = "objects";
//                             mobjectorig.id = i + 900;
//                             mobjectorig.type = 1;   // Cube
//                             mobjectorig.action = 0; // add/modify
//                             mobjectorig.scale.x = 0.5;
//                             mobjectorig.scale.y = 0.5;
//                             mobjectorig.scale.z = 0.1;
//                             myQuaternion.setRPY(0, 0, FuseObjects[i].object_rect3d_world.heading);

//                             mobjectorig.pose.orientation.w = myQuaternion.getW();
//                             mobjectorig.pose.orientation.x = myQuaternion.getX();
//                             mobjectorig.pose.orientation.y = myQuaternion.getY();
//                             mobjectorig.pose.orientation.z = myQuaternion.getZ();
//                             mobjectorig.lifetime = ros::Duration(0.2);
//                             mobjectorig.frame_locked = false;

//                             mobjectorig.pose.position.x = ball.x;
//                             mobjectorig.pose.position.y = ball.y;
//                             mobjectorig.pose.position.z = 0;
//                             mobjectorig.color.r = 1;
//                             mobjectorig.color.g = 0;
//                             mobjectorig.color.b = 0;
//                             mobjectorig.color.a = 0.6;
//                             visualization_msgs::Marker mtext;

//                             mtext.header.frame_id = "rslidar";
//                             mtext.ns = "rslidar";
//                             // set marker type
//                             mtext.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
//                             mtext.pose.orientation.w = 1.0; //文字的方向
//                             mtext.id =
//                                 i + 100; //用来标记同一帧不同的对象，如果后面的帧的对象少于前面帧的对象，那么少的id将在rviz中残留，所以需要后续的实时更新程序
//                             mtext.scale.x = 1.5;
//                             mtext.scale.y = 1.5;
//                             mtext.scale.z = 0.8; //文字的大小
//                             mtext.lifetime = ros::Duration(0.2);
//                             mtext.frame_locked = false;
//                             mtext.color.b = 25;
//                             mtext.color.g = 25;
//                             mtext.color.r = 240; //文字的颜色

//                             mtext.color.a = 1;
//                             geometry_msgs::Pose pose;
//                             pose.position.x = ball.x;
//                             pose.position.y = ball.y;
//                             pose.position.z = 0;
//                             mtext.text =
//                                 "id:" + std::to_string(FuseObjects[i].object_id) + " conf:" + std::to_string(FuseObjects[i].object_obj_confidence); //文字内容
//                             mtext.pose = pose;

//                         } else {
//                             visualization_msgs::Marker mobject2;
//                             mobject2.header.stamp = ros::Time::now();
//                             mobject2.header.frame_id = "rslidar";
//                             mobject2.ns = "objects";
//                             mobject2.id = 25 * i + j;
//                             mobject2.type = 1;   // Cube
//                             mobject2.action = 0; // add/modify

//                             myQuaternion.setRPY(0, 0, FuseObjects[i].object_rect3d_world.heading);

//                             mobject2.pose.orientation.w = myQuaternion.getW();
//                             mobject2.pose.orientation.x = myQuaternion.getX();
//                             mobject2.pose.orientation.y = myQuaternion.getY();
//                             mobject2.pose.orientation.z = myQuaternion.getZ();
//                             mobject2.scale.x = 0.3;
//                             mobject2.scale.y = 0.3;
//                             mobject2.scale.z = 0.1;

//                             mobject2.lifetime = ros::Duration(0.2);
//                             mobject2.frame_locked = false;

//                             mobject2.pose.position.x = precicted_msg.predicted_objects[a].predicted_trajs[b].predicted_traj_points[c].x;
//                             mobject2.pose.position.y = precicted_msg.predicted_objects[a].predicted_trajs[b].predicted_traj_points[c].y;
//                             mobject2.pose.position.z = 0;
//                             mobject2.color.r = 0;
//                             mobject2.color.g = 1;
//                             mobject2.color.b = 0;
//                             mobject2.color.a = 0.7;

//                             //原始位置
//                             ROS_INFO("Vusion_Publish prediction Info: obj_id:%d point_idx:%d  point_x:%f point_y:%f", mobject2.id, j, mobject2.pose.position.x,
//                                      mobject2.pose.position.y);
//                             marker_array_pgp.markers.push_back(mobject2);
//                         }
//                     }
//                 }
//             }

//             marker_array_pgp.markers.push_back(mobjectorig);
//             marker_array_pgp.markers.push_back(mtext);
//         }
//     }

//     msg_pub.publish(precicted_msg);
//     Vus_msg_pub1.publish(marker_array_pgp);
//     // r.sleep();
// }
// int main(int argc, char** argv) {
//     Objectlist(objectlist);

//     ros::init(argc, argv, "prediction");

//     Prediction prediction;

//     ros::spin();
//     return 0;
// }