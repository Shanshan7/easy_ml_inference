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

#include <json/json.h>
// #include <perception_msgs/FuseObjectList.h>

#include "JoysonPredictionCommon.h"
#include "JoysonHDMapCommon.h"


struct GlobalPose
{
    hdmap::Point pt;
    float heading;
};

/** 
 * @brief algorithm map input
 */
struct MapRepresentation
{
    typedef Eigen::Matrix<float, 1, -1, -1, 6, Eigen::RowMajor> lane_node_feats; // [batch_size, max_nodes, max_poses, node_feat_size]
    typedef Eigen::Matrix<float, 1, -1, -1, 6, Eigen::RowMajor> lane_node_masks; // [batch_size, max_nodes, max_poses, node_feat_size]
    typedef Eigen::Matrix<float, 1, -1, 15, Eigen::RowMajor> s_next; // [batch_size, max_nodes, traversal_horizon]
    typedef Eigen::Matrix<float, 1, -1, 15, Eigen::RowMajor> edge_type; // [batch_size, max_nodes, traversal_horizon]
};

// /** 
//  * @brief algorithm surrounding agent
//  */
// struct SurroundingAgentRepresentation
// {
//     typedef Eigen::Matrix<float, 1, -1, 5, 5, Eigen::RowMajor> surrounding_vehicles; // [batch_size, max_vehicles, t_h, nbr_feat_size] 
//     typedef Eigen::Matrix<float, 1, -1, 5, 5, Eigen::RowMajor> surrounding_vehicles_masks; // [batch_size, max_vehicles, t_h, nbr_feat_size] 
//     typedef Eigen::Matrix<float, 1, -1, 5, 5, Eigen::RowMajor> surrounding_pedsetrains; // [batch_size, max_peds, t_h, nbr_feat_size]
//     typedef Eigen::Matrix<float, 1, -1, 5, 5, Eigen::RowMajor> surrounding_pedsetrains_masks; // [batch_size, max_peds, t_h, nbr_feat_size] 
// };

struct SurroundingAgentRepresentation
{
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>>  surrounding_vehicles; // [batch_size, max_vehicles, t_h, nbr_feat_size] 
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> surrounding_vehicles_masks; // [batch_size, max_vehicles, t_h, nbr_feat_size] 
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> surrounding_pedsetrains; // [batch_size, max_peds, t_h, nbr_feat_size]
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> surrounding_pedsetrains_masks; // [batch_size, max_peds, t_h, nbr_feat_size] 
};

/** 
 * @brief algorithm surrounding agent mask
 * @details Returns key/val masks for agent-node attention layers. All agents except those within a distance threshold of
 *          the lane node are masked. The idea is to incorporate local agent context at each lane node.
 */
struct AgentNodeMasks
{
    typedef Eigen::Matrix<float, 1, -1, -1, Eigen::RowMajor> vehicle_node_mask; // [batch_size, max_nodes, max_vehicles]
    typedef Eigen::Matrix<float, 1, -1, -1, Eigen::RowMajor> ped_node_masks; // [batch_size, max_nodes, max_pedestrians]
};

struct PredictionNetInput
{
    typedef Eigen::Matrix<float, 1, 5, 5, Eigen::RowMajor> target_agent_feats; // [batch_size, t_h, target_agent_feat_size]
    MapRepresentation map_representation;
    SurroundingAgentRepresentation surrounding_agent_representation;
    AgentNodeMasks agent_node_masks;
    typedef Eigen::Matrix<float, 1, -1, Eigen::RowMajor> init_node;
};

class JoysonPredictionPre
{
private:
    /* data */
private:
    /* function */
    // void GetTargetAgentGlobalPose(std::vector<perception_msgs::FuseObject> &fuse_objects, 
    //                               std::vector<GlobalPose> &global_pose_list);
    
    // void GetMapRepresentation(hdmap::LaneInfo &lane_info,
    //                           hdmap::CrosswalkInfo &crosswalk_info,
    //                           hdmap::StopLineInfo &stop_line_info,
    //                           std::vector<GlobalPose> &global_pose_list,
    //                           MapRepresentation &map_representation);
    
    // void GetSurroundingAgentRepresentation(std::vector<perception_msgs::FuseObject> &fuse_objects,
    //                                        SurroundingAgentRepresentation &surrounding_agent_representation);

    // void GetTargetAgentRepresentation(std::vector<perception_msgs::FuseObject> &fuse_objects,
    //                                   Eigen::Matrix<float, 1, 5, 5, Eigen::RowMajor> target_agent_feats);

    // void GetAgentNodeMasks(MapRepresentation &map_representation,
    //                        SurroundingAgentRepresentation &surrounding_agent_representation,
    //                        AgentNodeMasks &agent_node_masks);

    // void GetInitialNode(MapRepresentation &map_representation,
    //                     Eigen::Matrix<float, 1, -1, Eigen::RowMajor> init_node);

    // /*
    //     Generates vector HD map representation in the agent centric frame of reference
    //     :param origin: (x, y, yaw) of target agent in global co-ordinates
    //     :param lanes: lane centerline poses in global co-ordinates
    //     :param polygons: stop-line and cross-walk polygons in global co-ordinates
    //     :return:
    // */
    // void GetLaneNodeFeats(hdmap::LaneInfo &lane_info,
    //                       hdmap::CrosswalkInfo &crosswalk_info,
    //                       hdmap::StopLineInfo &stop_line_info,
    //                       MapRepresentation &map_representation);

public:
    JoysonPredictionPre(/* args */);
    ~JoysonPredictionPre();

    // void GetInputs(hdmap::LaneInfo &lane_info,
    //                hdmap::CrosswalkInfo &crosswalk_info,
    //                hdmap::StopLineInfo &stop_line_info,
    //                std::vector<perception_msgs::FuseObject> &fuse_objects, 
    //                PredictionNetInput prediction_net_input);

    int GetInputs(std::string data_pickle_path, PredictionNetInput &prediction_net_input);
};
