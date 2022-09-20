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
// #define _CRT_SECURE_NO_DEPRECATE 
#pragma once

#include<iostream>
#include<string>
# include<vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include<list>
#include <cstring>
#include <algorithm>
#include <cmath>
#include<map>
#include <Eigen/StdVector>
#include <perception_msgs/FuseObjectList.h>
#include <JoysonPredictionPre.h>
// #define M_PI       3.14159265358979323846;


// 获得障碍物的字典形式的数据
void GetDictandNum(perception_msgs::FuseObjectList &fuse_objects, int &vhicle_num,int &pedestrian_num,
                      std::map<uint16_t,std::map<uint64_t,Eigen::MatrixXf>,std::less<uint16_t>> &vhicle_out_maps,
                    std::map<uint16_t,std::map<uint64_t,Eigen::MatrixXf>,std::less<uint16_t>> &ped_out_maps)
{
    // vhicles map
    // 内层map {timestamp:Matrix,std::less(int)}  //Matrix:(x,y,vel,acc,heading,id_type)
    std::map<uint64_t,Eigen::MatrixXf,std::less<uint64_t>> vhicles_inter_maps;
    // 外层map {track_id:map,less(int)}   //map为内层的map根据strack_id冲小到大排序
    std::map<uint16_t,std::map<uint64_t,Eigen::MatrixXf>,std::less<uint16_t>> vhicles_out_maps;
    
    // pedestrians map
    // 内层map {timestamp:Matrix,std::less(int)}  //Matrix:(x,y,vel,acc,heading,id_type)
    std::map<uint64_t,Eigen::MatrixXf,std::less<uint64_t>> pedestrian_inter_maps;
    // 外层map {track_id:map,less(int)}   //map为内层的map根据strack_id冲小到大排序
    std::map<uint16_t,std::map<uint64_t,Eigen::MatrixXf>,std::less<uint16_t>> pedestrian_out_maps;
    

    // 得到objects
    // 创建一个ID的vector 所有的fus_object
    std::vector<uint16_t> v(fuse_objects.fuse_obj_num);
    // 初始化vhicle 和pedestrain的数量
    int max_vehicles=0;
    int max_pedestrians=0;
    for (int j=0;j<(int)(fuse_objects.fuse_obj_num);j++)
    {
        uint16_t object_id = fuse_objects.FuseObjects[j].object_id;
        if (std::count(v.begin(), v.end(), object_id)){continue;}
        v[j]=object_id;
        if(fuse_objects.FuseObjects[j].object_type==4){max_vehicles++;}
        else if (fuse_objects.FuseObjects[j].object_type==2){max_pedestrians++;}
        
        for (int i=0;i<(int)(fuse_objects.fuse_obj_num);i++)
        {
            if ((object_id == fuse_objects.FuseObjects[i].object_id)&&fuse_objects.FuseObjects[i].object_type==4)
            {
            Eigen::MatrixXf obj_pose(1,5);
            // float obj_vel = fuse_objects.FuseObjects[i].object_velocity.norm();
            float obj_vel = sqrt(fuse_objects.FuseObjects[i].object_velocity.x*fuse_objects.FuseObjects[i].object_velocity.x+fuse_objects.FuseObjects[i].object_velocity.y*fuse_objects.FuseObjects[i].object_velocity.y);
            float obj_acc = sqrt(fuse_objects.FuseObjects[i].object_acc.x*fuse_objects.FuseObjects[i].object_acc.x+fuse_objects.FuseObjects[i].object_acc.y*fuse_objects.FuseObjects[i].object_acc.y);
            float obj_heading = fuse_objects.FuseObjects[i].object_rect3d.heading;
            float type_id = (float)fuse_objects.FuseObjects[i].object_type;
            obj_pose<<fuse_objects.FuseObjects[i].object_rect3d.center.x,
                        fuse_objects.FuseObjects[i].object_rect3d.center.y,
                        obj_vel,obj_acc,obj_heading;

            vhicles_inter_maps[fuse_objects.FuseObjects[i].detected_time]=obj_pose;
            }
            else if ((object_id == fuse_objects.FuseObjects[i].object_id)&&fuse_objects.FuseObjects[i].object_type==2)
            {
            Eigen::MatrixXf obj_pose(1,5);
            float obj_vel =sqrt(fuse_objects.FuseObjects[i].object_velocity.x*fuse_objects.FuseObjects[i].object_velocity.x+fuse_objects.FuseObjects[i].object_velocity.y*fuse_objects.FuseObjects[i].object_velocity.y);
            float obj_acc =sqrt(fuse_objects.FuseObjects[i].object_acc.x*fuse_objects.FuseObjects[i].object_acc.x+fuse_objects.FuseObjects[i].object_acc.y*fuse_objects.FuseObjects[i].object_acc.y);
            float obj_heading = fuse_objects.FuseObjects[i].object_rect3d.heading;
            float type_id = (float)fuse_objects.FuseObjects[i].object_type;
            obj_pose<<fuse_objects.FuseObjects[i].object_rect3d.center.x,
                        fuse_objects.FuseObjects[i].object_rect3d.center.y,
                        obj_vel,obj_acc,obj_heading;

            pedestrian_inter_maps[fuse_objects.FuseObjects[i].detected_time]=obj_pose;
            }
        }
        if(fuse_objects.FuseObjects[j].object_type==4)
        {
            vhicles_out_maps[object_id]=vhicles_inter_maps;
        }
        else if (fuse_objects.FuseObjects[j].object_type==2)
        {
            pedestrian_out_maps[object_id]=pedestrian_inter_maps;
        }
    }
    // 这里可以返回max_vhicle,max_pedestrian,vhicles_out_maps,pedestrian_out_maps
    vhicle_num = max_vehicles;
    pedestrian_num=max_pedestrians;
    vhicle_out_maps=vhicles_out_maps;
    ped_out_maps=pedestrian_out_maps;
}

// 外部maps的keys处理
std::vector<uint16_t> OutKeySet(std::map<uint16_t,std::map<uint64_t,Eigen::MatrixXf>,std::less<uint16_t>>  test)
{
    std::vector<uint16_t> keys;
    for(std::map<uint16_t,std::map<uint64_t,Eigen::MatrixXf>,std::less<uint16_t>> ::iterator it = test.begin(); it != test.end(); ++it)
    {
        keys.push_back(it->first);
        // cout<<it->first<<endl;
    }
    return keys;
}

void GetArray(perception_msgs::FuseObjectList &fuse_objects,std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> &vhi_arrs,
                std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> &vhi_maskss,
                std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> &pedest_arrs,
                std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> &pedest_maskss)
{
    
    int vhicle_num;
    int pedestrian_num;
    std::map<uint16_t,std::map<uint64_t,Eigen::MatrixXf>,std::less<uint16_t>> vhicle_out_maps;
    std::map<uint16_t,std::map<uint64_t,Eigen::MatrixXf>,std::less<uint16_t>> ped_out_maps;
    GetDictandNum(fuse_objects,vhicle_num,pedestrian_num,vhicle_out_maps,ped_out_maps);
    
    /*vhicle的矩阵init*/ 
    // Eigen::MatrixXf vhi_mat(5,5);
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> vhi_arr;
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> vhi_masks;
    // 外部maps的keys向量
    std::vector<uint16_t> outkeys;
    outkeys = OutKeySet(vhicle_out_maps);
    // 遍历矩阵
    for (int i=0;i<(int)(outkeys.size());i++)
    {
        Eigen::MatrixXf vhi_mat(5,5);
        Eigen::MatrixXf vhi_mask(5,5);
        vhi_mask.fill(1);
        std::map<uint64_t,Eigen::MatrixXf,std::less<uint64_t>> inter_maps = vhicle_out_maps[outkeys[i]];
        for (int j=0;j<(int)(inter_maps.size());j++)
        {
            for(std::map<uint64_t,Eigen::MatrixXf,std::less<uint64_t>>::iterator it = inter_maps.begin(); it != inter_maps.end(); ++it)
            {
                vhi_mat.row(j)<<(it->second);
                vhi_mask.row(j).fill(0);
            }

        }
    vhi_arr[i]=vhi_mat;
    vhi_masks[i]=vhi_mask;
    }
      /*pedestrain的矩阵init*/ 
    // Eigen::MatrixXf pedest_mat(5,5);pedest_masks;
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> pedest_arr;
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> pedest_masks;
    // 外部maps的keys向量
    outkeys = OutKeySet(ped_out_maps);
    // 遍历矩阵
    for (int i=0;i<(int)(outkeys.size());i++)
    {
        Eigen::MatrixXf pedest_mat(5,5);
        Eigen::MatrixXf pedest_mask(5,5);
        pedest_mask.fill(1);
        std::map<uint64_t,Eigen::MatrixXf,std::less<uint64_t>> inter_maps = ped_out_maps[outkeys[i]];
        for (int j=0;j<(int)(inter_maps.size());j++)
        {
            for(std::map<uint64_t,Eigen::MatrixXf,std::less<uint64_t>>::iterator it = inter_maps.begin(); it != inter_maps.end(); ++it)
            {
                pedest_mat.row(j)<<(it->second);
                pedest_mask.row(j).fill(0);
            }

        }
    pedest_arr[i]=pedest_mat;
    pedest_masks[i]=pedest_mask;
    }
    vhi_arrs = vhi_arr;
    vhi_maskss = vhi_masks;
    pedest_arrs = pedest_arr;
    pedest_maskss=pedest_masks;
}

// 获得矩阵
void GetSurroundingAgentRepresentation(perception_msgs::FuseObjectList &fuse_objects,
        SurroundingAgentRepresentation &surrounding_agent_representation)
{
        GetArray(fuse_objects, surrounding_agent_representation.surrounding_vehicles,
            surrounding_agent_representation.surrounding_vehicles_masks,
            surrounding_agent_representation.surrounding_pedsetrains,
            surrounding_agent_representation.surrounding_pedsetrains_masks);
}

