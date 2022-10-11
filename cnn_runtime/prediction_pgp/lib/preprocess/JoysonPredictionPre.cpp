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
	// memset(prediction_net_input, 0, sizeof(prediction_net_input));
	// inititial prediction input
	prediction_net_input.target_agent_feats.resize(5,5);

	prediction_net_input.map_representation.lane_node_feats = Eigen::Tensor<float, 3>(164,20,6);
	prediction_net_input.map_representation.lane_node_masks = Eigen::Tensor<float, 3>(164,20,6);
	prediction_net_input.map_representation.s_next.resize(164,15);
	prediction_net_input.map_representation.edge_type.resize(164,15);

	prediction_net_input.surrounding_agent_representation.surrounding_vehicles = Eigen::Tensor<float, 3>(84,5,5);
	prediction_net_input.surrounding_agent_representation.surrounding_vehicles_masks = Eigen::Tensor<float, 3>(84,5,5);
	prediction_net_input.surrounding_agent_representation.surrounding_pedsetrains = Eigen::Tensor<float, 3>(77,5,5);
	prediction_net_input.surrounding_agent_representation.surrounding_pedsetrains_masks = Eigen::Tensor<float, 3>(77,5,5);

	prediction_net_input.agent_node_masks.vehicle_node_mask.resize(164,84);
	prediction_net_input.agent_node_masks.ped_node_masks.resize(164,77);

	prediction_net_input.init_node.resize(164);

	Json::Reader reader;
	Json::Value root;

	// 从文件中读取，保证当前文件有test.json文件
	std::ifstream in(data_pickle_path, std::ios::binary);
	//in.open("test.json", ios::binary);

	if( !in.is_open() )  
	{ 
		std::cout << "Error opening file\n"; 
		return -1; 
	}

	if(reader.parse(in,root))
	{
		// 读取根节点信息
		// string inputs = root["inputs"].asString();
		// string ground_truth = root["ground_truth"].asString();

		// cout << "Inputs is " << root["inputs"]["instance_token"]<< endl;
		// cout << "ground_truth " << ground_truth << endl;

		// 读取子节点信息
		// inputs
		std::string instance_token = root["inputs"]["instance_token"].asString();
		std::string sample_token = root["inputs"]["sample_token"].asString();
		// 读取数组信息
		std::cout << "reading info: " << std::endl;
		// 输出信息
		std::cout<<"instance_token " << instance_token << std::endl;
		std::cout<<"sample_token " << sample_token << std::endl;
		// 输出信息
		// 地图的相关信息
		// lane_node_feats
		int lane_node_feats_c = root["inputs"]["map_representation"]["lane_node_feats"].size();
		int lane_node_feats_h = root["inputs"]["map_representation"]["lane_node_feats"][0].size();
		int lane_node_feats_w = root["inputs"]["map_representation"]["lane_node_feats"][0][0].size();
		for (int i = 0; i < lane_node_feats_c; i++)
		{
			for (int j = 0; j < lane_node_feats_h; j++)
			{
				for (int k = 0; k < lane_node_feats_w; k++)
				{
					prediction_net_input.map_representation.lane_node_feats(i*lane_node_feats_h*lane_node_feats_w+j*lane_node_feats_w+k) = 
					      root["inputs"]["map_representation"]["lane_node_feats"][i][j][k].asDouble();
					
				}
			}
		}

		// lane_node_masks
		int lane_node_masks_c = root["inputs"]["map_representation"]["lane_node_masks"].size();
		int lane_node_masks_h = root["inputs"]["map_representation"]["lane_node_masks"][0].size();
		int lane_node_masks_w = root["inputs"]["map_representation"]["lane_node_masks"][0][0].size();
		for (int i = 0; i < lane_node_masks_c; i++)
		{
			for (int j = 0; j < lane_node_masks_h; j++)
			{
				for (int k = 0; k < lane_node_masks_w; k++)
				{
					prediction_net_input.map_representation.lane_node_feats(i*lane_node_masks_h*lane_node_masks_w+j*lane_node_masks_w+k) = 
					      root["inputs"]["map_representation"]["lane_node_masks"][i][j][k].asDouble();
				}
			}
		}

		// s_next
		int s_next_h = root["inputs"]["map_representation"]["s_next"].size();
		int s_next_w = root["inputs"]["map_representation"]["s_next"][0].size();
		for (int i = 0; i < s_next_h; i++)
		{
			for (int j = 0; j < s_next_w; j++)
			{
				prediction_net_input.map_representation.s_next(i*s_next_w+j) = 
					      root["inputs"]["map_representation"]["s_next"][i][j].asDouble();	
			}
		}

		// edge_type
		int edge_type_h = root["inputs"]["map_representation"]["edge_type"].size();
		int edge_type_w = root["inputs"]["map_representation"]["edge_type"][0].size();
		for (int i = 0; i < edge_type_h; i++)
		{
			for (int j = 0; j < edge_type_w; j++)
			{
				prediction_net_input.map_representation.edge_type(i*edge_type_w+j) = 
					      root["inputs"]["map_representation"]["edge_type"][i][j].asDouble();
			}
		}
		
		// 周边障碍物的相关信息
		// vehicle_masks
		int vehicle_masks_c = root["inputs"]["surrounding_agent_representation"]["vehicle_masks"].size();
		int vehicle_masks_h = root["inputs"]["surrounding_agent_representation"]["vehicle_masks"][0].size();
		int vehicle_masks_w = root["inputs"]["surrounding_agent_representation"]["vehicle_masks"][0][0].size();
		for (int i = 0; i < vehicle_masks_c; i++)
		{
			for (int j = 0; j < vehicle_masks_h; j++)
			{
				for (int k = 0; k < vehicle_masks_w; k++)
				{
					prediction_net_input.surrounding_agent_representation.surrounding_vehicles(i*vehicle_masks_h*vehicle_masks_w+j*vehicle_masks_w+k) = 
					      root["inputs"]["surrounding_agent_representation"]["vehicle_masks"][i][j][k].asDouble();
				}			
			}
		}

		// vehicle
		int vehicles_c = root["inputs"]["surrounding_agent_representation"]["vehicles"].size();
		int vehicles_h = root["inputs"]["surrounding_agent_representation"]["vehicles"][0].size();
		int vehicles_w = root["inputs"]["surrounding_agent_representation"]["vehicles"][0][0].size();
		for (int i = 0; i < vehicles_c; i++)
		{
			for (int j = 0; j < vehicles_h; j++)
			{
				for (int k = 0; k < vehicles_w; k++)
				{
					prediction_net_input.surrounding_agent_representation.surrounding_vehicles(i*vehicles_h*vehicles_w+j*vehicles_w+k) = 
					      root["inputs"]["surrounding_agent_representation"]["vehicles"][i][j][k].asDouble();
				}			
			}
		}

		// pedestrians
		int pedestrians_c = root["inputs"]["surrounding_agent_representation"]["pedestrians"].size();
		int pedestrians_h = root["inputs"]["surrounding_agent_representation"]["pedestrians"][0].size();
		int pedestrians_w = root["inputs"]["surrounding_agent_representation"]["pedestrians"][0][0].size();
		for (int i = 0; i < pedestrians_c; i++)
		{
			for (int j = 0; j < pedestrians_h; j++)
			{
				for (int k = 0; k < pedestrians_w; k++)
				{
					prediction_net_input.surrounding_agent_representation.surrounding_pedsetrains(i*pedestrians_h*pedestrians_w+j*pedestrians_w+k) = 
					      root["inputs"]["surrounding_agent_representation"]["pedestrians"][i][j][k].asDouble();
				}
			}
		}

		// pedestrian_masks
		int pedestrian_masks_c = root["inputs"]["surrounding_agent_representation"]["pedestrian_masks"].size();
		int pedestrian_masks_h = root["inputs"]["surrounding_agent_representation"]["pedestrian_masks"][0].size();
		int pedestrian_masks_w = root["inputs"]["surrounding_agent_representation"]["pedestrian_masks"][0][0].size();
		for (int i = 0; i < pedestrian_masks_c; i++)
		{
			for (int j = 0; j < pedestrian_masks_h; j++)
			{
				for (int k = 0; k < pedestrian_masks_w; k++)
				{
					prediction_net_input.surrounding_agent_representation.surrounding_pedsetrains_masks(i*pedestrian_masks_h*pedestrian_masks_w+j*pedestrian_masks_w+k) = 
					      root["inputs"]["surrounding_agent_representation"]["pedestrian_masks"][i][j][k].asDouble();
				}		
			}
		}

		// target_agent_representation
		int target_agent_representation_h = root["inputs"]["target_agent_representation"].size();
		int target_agent_representation_w = root["inputs"]["target_agent_representation"][0].size();
		for (int i = 0; i < target_agent_representation_h; i++)
		{
			for (int j = 0; j < target_agent_representation_w; j++)
			{
				prediction_net_input.target_agent_feats(i*target_agent_representation_w+j) = 
					      root["inputs"]["target_agent_representation"][i][j].asDouble();
			}
		}

		// angent_pedestrian
		int agent_pedestrian_h = root["inputs"]["agent_node_masks"]["pedestrians"].size();
		int agent_pedestrian_w = root["inputs"]["agent_node_masks"]["pedestrians"][0].size();
		for (int i = 0; i < agent_pedestrian_h; i++)
		{
			for (int j = 0; j < agent_pedestrian_w; j++)
			{
				prediction_net_input.agent_node_masks.ped_node_masks(i*agent_pedestrian_w+j) = 
					root["inputs"]["agent_node_masks"]["pedestrians"][i][j].asDouble();
			}
		}

		// angent_vehicles
		int agent_vehicles_h = root["inputs"]["agent_node_masks"]["vehicles"].size();
		int agent_vehicles_w = root["inputs"]["agent_node_masks"]["vehicles"][0].size();
		for (int i = 0; i < agent_vehicles_h; i++)
		{
			for (int j = 0; j < agent_vehicles_w; j++)
			{
				prediction_net_input.agent_node_masks.vehicle_node_mask(i*agent_vehicles_w+j) = 
					root["inputs"]["agent_node_masks"]["vehicles"][i][j].asDouble();
			}
		}

		// init_node
		int init_node_w = root["inputs"]["init_node"].size();
		for (int i = 0; i < init_node_w; i++)
		{
			prediction_net_input.init_node(i) = root["inputs"]["init_node"][i].asDouble();
		}

		// ***************ground_truth*********************
		// node_seq_gt
		std::vector<double> node_seq_gt;
		for (int i = 0; i < root["inputs"]["node_seq_gt"].size(); i++)
		{

			node_seq_gt.push_back(root["inputs"]["node_seq_gt"][i].asDouble());
		}

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

		// cout<<"lane_node_feats"<<lane_node_feats<<endl;
		std::cout << std::endl;

		std::cout << "Reading Complete!" << std::endl;
	}
	else
	{
		std::cout << "parse error\n" << std::endl;	
	}

	in.close();

	return 0;
}

void JoysonPredictionPre::NetInputTransform(PredictionNetInput &prediction_net_input,
                                            std::vector<float*> &input_img_data)
{
	// target_agent_feats
	std::cout << prediction_net_input.target_agent_feats.size() << std::endl;
	input_img_data[0] = prediction_net_input.target_agent_feats.data();

	// map representation
	std::cout << prediction_net_input.map_representation.lane_node_masks.size() << std::endl;
	input_img_data[1] = prediction_net_input.map_representation.lane_node_masks.data();
	input_img_data[2] = prediction_net_input.map_representation.lane_node_feats.data();

	// surrounding agent representation
	std::cout << prediction_net_input.surrounding_agent_representation.surrounding_vehicles.size() << std::endl;
	std::cout << prediction_net_input.surrounding_agent_representation.surrounding_pedsetrains_masks.size() << std::endl;
	input_img_data[3] = prediction_net_input.surrounding_agent_representation.surrounding_vehicles.data();
	input_img_data[4] = prediction_net_input.surrounding_agent_representation.surrounding_vehicles_masks.data();
	input_img_data[5] = prediction_net_input.surrounding_agent_representation.surrounding_pedsetrains.data();
	input_img_data[6] = prediction_net_input.surrounding_agent_representation.surrounding_pedsetrains_masks.data();

	// surrounding agent mask
	std::cout << prediction_net_input.agent_node_masks.vehicle_node_mask.size() << std::endl;
	std::cout << prediction_net_input.agent_node_masks.ped_node_masks.size() << std::endl;
	input_img_data[7] = prediction_net_input.agent_node_masks.vehicle_node_mask.data();
	input_img_data[8] = prediction_net_input.agent_node_masks.ped_node_masks.data();

	// map representation
	std::cout << prediction_net_input.map_representation.s_next.size() << std::endl;
	input_img_data[9] = prediction_net_input.map_representation.s_next.data();
	input_img_data[10] = prediction_net_input.map_representation.edge_type.data();

	// init node
	// input_img_data[11] = prediction_net_input.init_node.data();
}