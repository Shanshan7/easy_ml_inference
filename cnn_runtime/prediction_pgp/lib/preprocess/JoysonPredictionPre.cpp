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
#include "JoysonPredictionPre.h"


JoysonPredictionPre::JoysonPredictionPre()
{

}

JoysonPredictionPre::~JoysonPredictionPre()
{

}

int JoysonPredictionPre::GetInputs(std::string data_pickle_path, PredictionNetInput &prediction_net_input)
{

	Json::Reader reader;
	Json::Value root;

	//从文件中读取，保证当前文件有test.json文件
	std::ifstream in(data_pickle_path, std::ios::binary);
	//in.open("test.json", ios::binary);

	if( !in.is_open() )  
	{ 
		std::cout << "Error opening file\n"; 
		return -1; 
	}

	if(reader.parse(in,root))
	{
	//读取根节点信息
	// string inputs = root["inputs"].asString();
    // string ground_truth = root["ground_truth"].asString();

	// cout << "Inputs is " << root["inputs"]["instance_token"]<< endl;
	// cout << "ground_truth " << ground_truth << endl;

	//读取子节点信息
    // inputs
	std::string instance_token = root["inputs"]["instance_token"].asString();
	std::string sample_token = root["inputs"]["sample_token"].asString();
	//读取数组信息
	std::cout << "reading info:" << std::endl;
	// 输出信息
	// std::cout<<"instance_token:  "<<instance_token<<std::endl;
	// std::cout<<"sample_token:  "<<sample_token<<std::endl;
	// std::cout<<"lane_node_feats:   "<<std::endl;
	// lane_node_feats
	std::vector<std::vector<std::vector<double>>> lane_node_feats;
	for (int i = 0; i < root["inputs"]["map_representation"]["lane_node_feats"].size(); i++)
	{
		std::vector<std::vector<double>> lane_feats_1;
		for (int j = 0; j < root["inputs"]["map_representation"]["lane_node_feats"][i].size(); j++)
		{
			std::vector<double> lane_feats_2;
			for (int k = 0; k < root["inputs"]["map_representation"]["lane_node_feats"][i][j].size(); k++)
			{
				lane_feats_2.push_back(root["inputs"]["map_representation"]["lane_node_feats"][i][j][k].asDouble());
				
			}
			lane_feats_1.push_back(lane_feats_2);
		}
		lane_node_feats.push_back(lane_feats_1);
	}

	// lane_node_masks
	std::vector<std::vector<std::vector<double>>> lane_node_masks;
	for (int i = 0; i < root["inputs"]["map_representation"]["lane_node_masks"].size(); i++)
	{
		std::vector<std::vector<double>> lane_node_masks_1;
		for (int j = 0; j < root["inputs"]["map_representation"]["lane_node_masks"][i].size(); j++)
		{
			std::vector<double> lane_node_masks_2;
			for (int k = 0; k < root["inputs"]["map_representation"]["lane_node_masks"][i][j].size(); k++)
			{
				lane_node_masks_2.push_back(root["inputs"]["map_representation"]["lane_node_masks"][i][j][k].asDouble());
			}
			lane_node_masks_1.push_back(lane_node_masks_2);
			
		}
		lane_node_masks.push_back(lane_node_masks_1);
	}
	// s_next
	std::vector<std::vector<double>> s_next;
	for (int i = 0; i < root["inputs"]["map_representation"]["s_next"].size(); i++)
	{
		std::vector<double> s_next_1;
		for (int j = 0; j < root["inputs"]["map_representation"]["s_next"][i].size(); j++)
		{
		
			s_next_1.push_back(root["inputs"]["map_representation"]["s_next"][i][j].asDouble());
			
		}
		s_next.push_back(s_next_1);
	}

	// edge_type
	std::vector<std::vector<double>> edge_type;
	for (int i = 0; i < root["inputs"]["map_representation"]["edge_type"].size(); i++)
	{
		std::vector<double> edge_type_1;
		for (int j = 0; j < root["inputs"]["map_representation"]["edge_type"][i].size(); j++)
		{
		
			edge_type_1.push_back(root["inputs"]["map_representation"]["edge_type"][i][j].asDouble());
			
		}
		edge_type.push_back(edge_type_1);
	}
	
	// vehicle_masks
	std::vector<std::vector<std::vector<double>>> vehicle_masks;
	for (int i = 0; i < root["inputs"]["surrounding_agent_representation"]["vehicle_masks"].size(); i++)
	{
		std::vector<std::vector<double>> vehicle_masks_1;
		for (int j = 0; j < root["inputs"]["surrounding_agent_representation"]["vehicle_masks"][i].size(); j++)
		{
			std::vector<double> vehicle_masks_2;
			for (int k = 0; k < root["inputs"]["surrounding_agent_representation"]["vehicle_masks"][i][j].size(); k++)
			{
				vehicle_masks_2.push_back(root["inputs"]["surrounding_agent_representation"]["vehicle_masks"][i][j][k].asDouble());
			}
			vehicle_masks_1.push_back(vehicle_masks_2);
			
		}
		vehicle_masks.push_back(vehicle_masks_1);
	}

	// vehicle
	std::vector<std::vector<std::vector<double>>> vehicle;
	for (int i = 0; i < root["inputs"]["surrounding_agent_representation"]["vehicles"].size(); i++)
	{
		std::vector<std::vector<double>> vehicle_1;
		for (int j = 0; j < root["inputs"]["surrounding_agent_representation"]["vehicles"][i].size(); j++)
		{
			std::vector<double> vehicle_2;
			for (int k = 0; k < root["inputs"]["surrounding_agent_representation"]["vehicles"][i][j].size(); k++)
			{
				vehicle_2.push_back(root["inputs"]["surrounding_agent_representation"]["vehicles"][i][j][k].asDouble());
			}
			vehicle_1.push_back(vehicle_2);
			
		}
		vehicle.push_back(vehicle_1);
	}

	// pedestrians
	std::vector<std::vector<std::vector<double>>> pedestrians;
	for (int i = 0; i < root["inputs"]["surrounding_agent_representation"]["pedestrians"].size(); i++)
	{
		std::vector<std::vector<double>> pedestrians_1;
		for (int j = 0; j < root["inputs"]["surrounding_agent_representation"]["pedestrians"][i].size(); j++)
		{
			std::vector<double> pedestrians_2;
			for (int k = 0; k < root["inputs"]["surrounding_agent_representation"]["pedestrians"][i][j].size(); k++)
			{
				pedestrians_2.push_back(root["inputs"]["surrounding_agent_representation"]["pedestrians"][i][j][k].asDouble());
			}
			pedestrians_1.push_back(pedestrians_2);
			
		}
		pedestrians.push_back(pedestrians_1);
	}

	// pedestrian_masks
	std::vector<std::vector<std::vector<double>>> pedestrian_masks;
	for (int i = 0; i < root["inputs"]["surrounding_agent_representation"]["pedestrian_masks"].size(); i++)
	{
		std::vector<std::vector<double>> pedestrian_masks_1;
		for (int j = 0; j < root["inputs"]["surrounding_agent_representation"]["pedestrian_masks"][i].size(); j++)
		{
			std::vector<double> pedestrian_masks_2;
			for (int k = 0; k < root["inputs"]["surrounding_agent_representation"]["pedestrian_masks"][i][j].size(); k++)
			{
				pedestrian_masks_2.push_back(root["inputs"]["surrounding_agent_representation"]["pedestrian_masks"][i][j][k].asDouble());
			}
			pedestrian_masks_1.push_back(pedestrian_masks_2);
			
		}
		pedestrian_masks.push_back(pedestrian_masks_1);
	}

	// target_agent_representation
	std::vector<std::vector<double>> target_agent_representation;
	for (int i = 0; i < root["inputs"]["target_agent_representation"].size(); i++)
	{
		std::vector<double> target_agent_representation_1;
		for (int j = 0; j < root["inputs"]["target_agent_representation"][i].size(); j++)
		{
			// std::vector<double> target_agent_representation_2;
			// for (int k = 0; k < root["inputs"]["surrounding_agent_representation"]["target_agent_representation"][i][j].size(); k++)
			// {
			// 	target_agent_representation_2.push_back(root["inputs"]["surrounding_agent_representation"]["target_agent_representation"][i][j][k].asDouble());
			// }
			target_agent_representation_1.push_back(root["inputs"]["target_agent_representation"][i][j].asDouble());
			
		}
		target_agent_representation.push_back(target_agent_representation_1);
	}

	// angent_pedestrian
	std::vector<std::vector<double>> agent_node_masks_pedestrian;
	for (int i = 0; i < root["inputs"]["agent_node_masks"]["pedestrians"].size(); i++)
	{

		std::vector<double> agent_node_masks_pedestrian_2;
		for (int k = 0; k < root["inputs"]["agent_node_masks"]["pedestrians"][i].size(); k++)
		{
			agent_node_masks_pedestrian_2.push_back(root["inputs"]["agent_node_masks"]["pedestrians"][i][k].asDouble());
		}

		agent_node_masks_pedestrian.push_back(agent_node_masks_pedestrian_2);
	}

	// angent_vehicles
	std::vector<std::vector<double>> agent_node_masks_vehicles;
	for (int i = 0; i < root["inputs"]["agent_node_masks"]["vehicles"].size(); i++)
	{

		std::vector<double> agent_node_masks_vehicles_2;
		for (int k = 0; k < root["inputs"]["agent_node_masks"]["vehicles"][i].size(); k++)
		{
			agent_node_masks_vehicles_2.push_back(root["inputs"]["agent_node_masks"]["vehicles"][i][k].asDouble());
		}

		agent_node_masks_vehicles.push_back(agent_node_masks_vehicles_2);
	}

	// init_node
	std::vector<double> init_node;
	for (int i = 0; i < root["inputs"]["init_node"].size(); i++)
	{

		init_node.push_back(root["inputs"]["init_node"][i].asDouble());
	}
	// node_seq_gt
	std::vector<double> node_seq_gt;
	for (int i = 0; i < root["inputs"]["node_seq_gt"].size(); i++)
	{

		node_seq_gt.push_back(root["inputs"]["node_seq_gt"][i].asDouble());
	}

	// ***************ground_truth*********************
	// traj
	std::vector<std::vector<double>> traj;
	for (int i = 0; i < root["ground_truth"]["traj"].size(); i++)
	{
		std::vector<double> traj_2;
		for (int k = 0; k < root["ground_truth"]["traj"][i].size(); k++)
		{
			traj_2.push_back(root["ground_truth"]["traj"][i][k].asDouble());
		}

		traj.push_back(traj_2);
	}

	// evf_gt
	std::vector<std::vector<double>> evf_gt;
	for (int i = 0; i < root["ground_truth"]["evf_gt"].size(); i++)
	{

		std::vector<double> evf_gt_2;
		for (int k = 0; k < root["ground_truth"]["evf_gt"][i].size(); k++)
		{
			evf_gt_2.push_back(root["ground_truth"]["evf_gt"][i][k].asDouble());
		}

		evf_gt.push_back(evf_gt_2);
	}

	//读取数组信息
	std::cout << "reading info:" << std::endl;
	// 输出信息
	std::cout<<"instance_token"<<instance_token<<std::endl;
	std::cout<<"sample_token"<<sample_token<<std::endl;
	// cout<<"lane_node_feats"<<lane_node_feats<<endl;
	std::cout << std::endl;

	std::cout << "Reading Complete!" << std::endl;
	}
	// else
	// {
	// cout << "parse error\n" << endl;	
	// }

	in.close();

	return 0;
}

void JoysonPredictionPre::NetInputTransform(PredictionNetInput &prediction_net_input)
{
	// target_agent_feats
	inputImgData[0] = prediction_net_input.target_agent_feats.data();

	// map representation
	inputImgData[1] = prediction_net_input.map_representation.lane_node_feats.data();
	inputImgData[2] = prediction_net_input.map_representation.lane_node_masks.data();
	inputImgData[3] = prediction_net_input.map_representation.s_next.data();
	inputImgData[4] = prediction_net_input.map_representation.edge_type.data();

	// surrounding agent representation
	inputImgData[5] = prediction_net_input.surrounding_agent_representation.surrounding_vehicles.data();
	inputImgData[6] = prediction_net_input.surrounding_agent_representation.surrounding_vehicles_masks.data();
	inputImgData[7] = prediction_net_input.surrounding_agent_representation.surrounding_pedsetrains.data();
	inputImgData[8] = prediction_net_input.surrounding_agent_representation.surrounding_pedsetrains_masks.data();

	// surrounding agent mask
	inputImgData[9] = prediction_net_input.agent_node_masks.vehicle_node_mask.data();
	inputImgData[10] = prediction_net_input.agent_node_masks.ped_node_masks.data();

	// init node
	inputImgData[11] = prediction_net_input.init_node.data();
}