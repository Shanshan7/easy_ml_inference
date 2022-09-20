
// /******************************************************************************
// **                                                                           **
// ** Copyright (C) Joyson Electronics (2022)                                   **
// **                                                                           **
// ** All rights reserved.                                                      **
// **                                                                           **
// ** This document contains proprietary information belonging to Joyson        **
// ** Electronics. Passing on and copying of this document, and communication   **
// ** of its contents is not permitted without prior written authorization.     **
// **                                                                           **
// ******************************************************************************/
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <list>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <map>
#include <Eigen/StdVector>

#include <localization_msgs/Localization.h>

// 全局转局部 1.angle_of_rotation;2.make_2d_rotation_matrix 3.coords

// angle_of_rotation
//sign
float sign(float x)
{
    if(x>0)
    {
        return 1;
    }
    else if(x==0)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}
float AngleofRotation(float yaw)
{
    float yaw_o = (M_1_PI/2)+sign(-yaw)*abs(yaw);
    return yaw_o;
}
// quaternion_yaw
float quaternion_yaw(Eigen::Quaterniond rotation)
{

    Eigen::Matrix3d R=rotation.toRotationMatrix();
    Eigen::Vector3d ra1=R.eulerAngles(2,1,0);
    float current_quan_yaw = atan2(ra1[1],ra1[0]);
    return current_quan_yaw;
}
Eigen::MatrixXf make_2d_rotation_matrix(float angle_in_radians)
{
    Eigen::MatrixXf rad_mat(2,2);
    rad_mat<<cos(angle_in_radians),-sin(angle_in_radians),
                sin(angle_in_radians),cos(angle_in_radians);
    return rad_mat;
}

Eigen::Matrix2f ConvertGlobalCoordstoLocal(Eigen::MatrixXf coordiantes,
                                            Eigen::Vector3f translation,
                                            Eigen::Quaterniond rotation)
{
    float quat_yaw = quaternion_yaw(rotation);
    float yaw = AngleofRotation(quat_yaw);
    Eigen::MatrixXf transform = make_2d_rotation_matrix(yaw);
    Eigen::Matrix2f trans;
    trans<<translation[0],translation[1];
    Eigen::MatrixXf one = Eigen::MatrixXf::Ones(coordiantes.rows(), coordiantes.cols());
    one.col(0)=one.col(0)*trans(0,0);
    one.col(1)=one.col(1)*trans(0,1);
    Eigen::Matrix2f coords = (coordiantes - one).transpose();

    return (transform*coords).transpose().leftCols(2);
}


void TargetAgentRepresentation(std::vector<localization_msgs::Localization> &fuse_objects,
                                   Eigen::MatrixXf &target_agent_featss)

{
    Eigen::MatrixXf target_agent_feats(5,5);
    int t_h=2;//风险点
    for (int i=0;i<2*t_h+1;i++)
    {
        Eigen::Matrix3f vel_mat;
        vel_mat<<fuse_objects[i].velocity.x,fuse_objects[i].velocity.y,fuse_objects[i].velocity.z;
        float vel= vel_mat.norm();
        Eigen::Matrix3f acc_mat;
        acc_mat<<fuse_objects[i].acceleration.x,fuse_objects[i].acceleration.y,fuse_objects[i].acceleration.z;
        float acc= acc_mat.norm();
        Eigen::Quaterniond heading;
        heading.w()=fuse_objects[i].rotation.w;
        heading.x()=fuse_objects[i].rotation.x;
        heading.y()=fuse_objects[i].rotation.y;
        heading.z()=fuse_objects[i].rotation.z;
        
        float yaw = quaternion_yaw(heading);
        Eigen::Vector3f translation;
        translation[0]=fuse_objects[i].translation.x;
        translation[1]=fuse_objects[i].translation.y;
        translation[2]=fuse_objects[i].translation.z;
        Eigen::Matrix2f coords;
        coords<<(float)translation[0],(float)translation[1];
        Eigen::Matrix2f local_xy;
        local_xy=ConvertGlobalCoordstoLocal(coords,translation,heading);
        target_agent_feats.row(5-i)<<local_xy(0,0),local_xy(0,1),vel,acc,yaw;

    }
    target_agent_featss = target_agent_feats;
}


 void GetTargetAgentRepresentation(std::vector<localization_msgs::Localization> &fuse_objects,
                                      Eigen::MatrixXf &target_agent_feats)
{
    // Localizations locat;
    // Eigen::MatrixXf target_agent_feats(5,5);
    TargetAgentRepresentation(fuse_objects,target_agent_feats);

}
//int main()
//{
//    // struct Localizations locas;
//    // Eigen::MatrixXf aa;
//    // aa=GetTargetAgentRepresentation(locas);
//    std::vector<localization_msgs::Localization> fuse_objects;
//    Eigen::MatrixXf target_agent_feats;
//    GetTargetAgentRepresentation(fuse_objects,
//                                  target_agent_feats);
//    return 0;
// }
