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
#include <vector>
#include <algorithm>
#include <map>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "JoysonPredictionCommon.h"

struct ClusterDataInfo
{
    Eigen::VectorXf cluser_cnts;
    Eigen::VectorXf cluster_lbls;
    Eigen::VectorXf cluster_ranks;
};

class ClusterFunc
{
public:
    int cluster_count;
    ClusterDataInfo cluster_data_info;
    Eigen::MatrixXd traj_clustered;

public:
    ClusterFunc();
    ~ClusterFunc();

    void ClusterTraj(int cluster_count, 
                    float *traj,
                    std::vector<int> traj_shape, 
                    std::vector<prediction::PredictedObject> &cluster_object_trajectory);
    void GetObjects();

private:
    void Cluster(int n_clusters, cv::Mat &data, cv::Mat &labels, cv::Mat &centers);
    void RankClusters(Eigen::MatrixXd &cluster_centers);
    void ClusterAndRank(int k, float *data);
    void ClusterUnique(int *cluster_labels, 
                       int cluster_labels_length);
    void RemoveRow(Eigen::MatrixXd& eigen_matrix, 
                   unsigned int row_to_remove);
    void RemoveRow(Eigen::VectorXf& eigen_vector, 
                   unsigned int row_to_remove);
    void DistCalculate(Eigen::MatrixXd &array1, 
                       Eigen::MatrixXd &array2, 
                       Eigen::MatrixXd &out_array);
    void RepeatImpl(Eigen::Tensor<float, 4> &input_eigen, 
                    std::vector<int> &repeat_shape,
                    Eigen::Tensor<float, 4> &output_eigen);
    void ScatterAddImpl(Eigen::Tensor<float, 4> &index_eigen,
                        Eigen::Tensor<float, 4> &src_eigen,
                        int add_dim,
                        Eigen::Tensor<float, 4> &output_eigen);

    std::map<int, int> cluster_unique_map;
    Eigen::VectorXf ranks;
    Eigen::VectorXf cluster_counts;
};