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
#include "cluster_func.h"


ClusterFunc::ClusterFunc()
{
    cluster_data_info.cluster_ranks.resize(cluster_count);
}

ClusterFunc::~ClusterFunc()
{

}

void ClusterFunc::Cluster(int n_clusters, cv::Mat &data, cv::Mat &labels, cv::Mat &centers)
{
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1);
    cv::kmeans(data, n_clusters, labels, criteria, 10, cv::KMEANS_PP_CENTERS, centers);
}

void ClusterFunc::RankClusters(Eigen::MatrixXd &cluster_centers)
{
    cluster_counts.resize(cluster_unique_map.size());

    int cluster_counts_index = 0;
    for(std::map<int, int>::iterator it = cluster_unique_map.begin(); it != cluster_unique_map.end(); ++it)
    {
        cluster_counts(cluster_counts_index) = it->second;
        cluster_counts_index++;
    }

    int num_clusters = cluster_counts.size();
    Eigen::VectorXf cluster_ids;
    cluster_ids.resize(num_clusters);
    for (int t = 0; t < num_clusters; t++)
    {
        cluster_ids(t) = t;
    }

    for(int i = num_clusters; i <= 0; i--)
    {
        Eigen::MatrixXd centroid_dists;
        DistCalculate(cluster_centers, cluster_centers, centroid_dists);

        Eigen::MatrixXd n1(num_clusters, num_clusters);
        for (int k = 0; k < num_clusters; k++)
        {
            n1.col(k) = cluster_centers;
        }

        Eigen::MatrixXd wts_molecule = n1 * n1.transpose();
        Eigen::MatrixXd wts_denominator = n1 + n1.transpose();
        Eigen::MatrixXd wts = wts_molecule.array() / wts_denominator.array();

        Eigen::MatrixXd diag_inf = Eigen::MatrixXd::Identity(num_clusters, num_clusters) * 100000000;
        Eigen::MatrixXd dists;
        dists = wts * centroid_dists + diag_inf;

        // Get clusters with min Ward distance and select cluster with fewer counts
        Eigen::MatrixXd::Index minRow, minCol;
	    double min = dists.minCoeff(&minRow, &minCol);

        int c = (cluster_counts(minRow) <= cluster_counts(minCol)) ? minRow : minCol;
        int c_ = (cluster_counts[minRow] <= cluster_counts[minCol]) ? minCol : minRow;

        // Assign rank i to selected cluster
        ranks(cluster_ids(c)) = i;

        // Merge clusters and update identity of merged cluster
        cluster_centers.row(c_) = (cluster_counts(c_) * cluster_centers.row(c_) + 
                               cluster_counts(c) * cluster_centers.row(c)) /
                               (cluster_counts(c_) + cluster_counts(c));
        cluster_counts(c_) += cluster_counts(c);

        // Discard merged cluster
        RemoveRow(cluster_ids, c);
        RemoveRow(cluster_centers, c);
        RemoveRow(cluster_counts, c);
    }
}

void ClusterFunc::ClusterAndRank(int k, float *data)
{
    cv::Mat cluster_labels;
    cv::Mat cluster_centers;
    cv::Mat data_mat(1000, 8, CV_32FC1, data);

    Cluster(k, data_mat, cluster_labels, cluster_centers);
    int cluster_labels_length = 1000;
    ClusterUnique((int*)cluster_labels.data, cluster_labels_length);

    int row = cluster_centers.rows;
    int col = cluster_centers.cols;
    Eigen::MatrixXd cluster_centers_eigen(row, col);
    cv::cv2eigen(cluster_centers, cluster_centers_eigen);
    RankClusters(cluster_centers_eigen);

    cv::cv2eigen(cluster_labels, cluster_data_info.cluster_lbls);
    cluster_data_info.cluster_ranks = ranks;
    cluster_data_info.cluser_cnts = cluster_counts;
}

void ClusterFunc::ClusterTraj(int cluster_count,
                              float *traj,
                              std::vector<int> traj_shape, 
                              std::vector<prediction::PredictedObject> &cluster_object_trajectory)
{
    int traj_eigen_size = traj_shape.size();
    auto mapped_t = Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::ColMajor>>(&traj[0], 
                                                                        traj_shape[0], traj_shape[1],
                                                                        traj_shape[2], traj_shape[3]);
    Eigen::Tensor<float, 4> traj_eigen = mapped_t;

    // Initialize output tensors
    int batch_size = traj_shape[0];
    int num_samples = traj_shape[1];
    int traj_len = traj_shape[2];

    // Down-sample traj along time dimension for faster clustering
    Eigen::array<Eigen::DenseIndex, 4> offsets = { 0, 0, 0, 0 };
    Eigen::array<Eigen::DenseIndex, 4> extents = { 1, 2, 2, 3 };
    Eigen::Tensor<float, 4> traj_eigen_slice = traj_eigen.slice(offsets, extents);

    int reshape_dims = traj_eigen_slice.dimension(2) * traj_eigen_slice.dimension(3);
    auto data = Eigen::Tensor<float, 3>(traj_eigen_slice.reshape(Eigen::array<int, 3>({ batch_size, num_samples, reshape_dims })));

    // Cluster and rank
    for (int k = 0; k < batch_size; k++)
    {
        Eigen::array<Eigen::DenseIndex, 3> offsets = { k, 0, 0 };
        Eigen::array<Eigen::DenseIndex, 3> extents = { k+1, data.dimension(1), data.dimension(2) };
        Eigen::Tensor<float, 3> data_slice = traj_eigen.slice(offsets, extents);
        // Eigen::TensorMap<Eigen::Tensor<float, 2, 1>> a(nums, 4, 3);
        // Eigen::TensorMap<Eigen::Tensor<float, 2, 1>> b(a.data(), 2, 6);
        float *data_res = data_slice.data();
        ClusterAndRank(10, data_res);
    }

    Eigen::Tensor<float, 4> cluster_lbls_eigen(1, 1000, 12, 2);
    Eigen::Tensor<float, 4> cluster_lbls = Eigen::TensorMap<Eigen::Tensor<float, 4>>(
                                 cluster_data_info.cluster_lbls.data(), 1, 1000, 1, 1);
    Eigen::Tensor<float, 4> cnt_eigen(1, 10, 12, 2);
    Eigen::Tensor<float, 4> cluser_cnts = Eigen::TensorMap<Eigen::Tensor<float, 4>>(
                                 cluster_data_info.cluser_cnts.data(), 1, 10, 1, 1);
    Eigen::Tensor<float, 4> output_eigen(1, 10, 12, 2);
    std::vector<int> repeat_shape = {1, 1, 12, 2};

    RepeatImpl(cluster_lbls, repeat_shape, cluster_lbls_eigen);
    ScatterAddImpl(cluster_lbls_eigen, traj_eigen, 1, output_eigen);
    RepeatImpl(cluser_cnts, repeat_shape, cnt_eigen);
    output_eigen = output_eigen / cnt_eigen;

    std::vector<float> scores;
    for(int j = 0; j < cluster_data_info.cluster_ranks.size(); j++)
    {
        scores.push_back(1.0 / cluster_data_info.cluster_ranks(j) / 
                         cluster_data_info.cluster_ranks.sum());
    }
}

void ClusterFunc::ClusterUnique(int *cluster_labels, 
                                int cluster_labels_length)
{
    // int cluster_labels_length = sizeof(cluster_labels) / sizeof(int);
    // std::cout << "cluster_labels_length: " << cluster_labels_length << std::endl;

	for (int i = 0; i < cluster_labels_length; i++)
	{	
		auto it = cluster_unique_map.find(cluster_labels[i]);	// 判断容器内是否有相同key值
		if (it==cluster_unique_map.end())
		{
			// std::cout << cluster_labels[i] << std::endl;
			cluster_unique_map.insert(std::make_pair(cluster_labels[i], 1));	// 如果没有则插入
		}
		else
		{
            it->second += 1;	// 如果有，则count+1
		}
	}
}

void ClusterFunc::DistCalculate(Eigen::MatrixXd &array1, 
                                Eigen::MatrixXd &array2, 
                                Eigen::MatrixXd &out_array)
{
    assert(array1.rows()==array2.rows() && array1.cols()==array2.cols());
    // assert('XA and XB must have the same number of columns ');

    int array1_row = array1.rows();
    int array2_row = array2.rows();
    out_array.resize(array1_row, array2_row);

    for (int i = 0; i < array1_row; i++)
    {
        for (int j = 0; j < array2_row; j++)
        {
            Eigen::VectorXd array1_row_ele;
            array1_row_ele.resize(array1_row);
            array1_row_ele = array1.row(i);

            Eigen::VectorXd array2_col_ele;
            array2_col_ele.resize(array2_row);
            array2_col_ele = array2.transpose().col(j);

            Eigen::VectorXd out_array_ele;
            out_array_ele.resize(array1_row);
            out_array_ele = array1_row_ele - array2_col_ele;

            out_array(i, j) = sqrt(abs(out_array_ele.array().square().sum()));
        }
    }
}

void ClusterFunc::RemoveRow(Eigen::MatrixXd& eigen_matrix, unsigned int row_to_remove) 
{
  unsigned int num_rows = eigen_matrix.rows() - 1;
  unsigned int num_cols = eigen_matrix.cols();
 
  if(row_to_remove < num_rows) {
    eigen_matrix.block(row_to_remove, 0, num_rows - row_to_remove, num_cols) =
        eigen_matrix.block(row_to_remove + 1, 0, num_rows - row_to_remove, num_cols);
  }
 
  eigen_matrix.conservativeResize(num_rows, num_cols);
}

void ClusterFunc::RemoveRow(Eigen::VectorXf& eigen_vector, unsigned int row_to_remove) 
{
  unsigned int num_rows = eigen_vector.rows() - 1;
  unsigned int num_cols = eigen_vector.cols();
 
  if(row_to_remove < num_rows) {
    eigen_vector.block(row_to_remove, 0, num_rows - row_to_remove, num_cols) =
        eigen_vector.block(row_to_remove + 1, 0, num_rows - row_to_remove, num_cols);
  }
 
  eigen_vector.conservativeResize(num_rows);
}

void ClusterFunc::RepeatImpl(Eigen::Tensor<float, 4> &input_eigen, std::vector<int> &repeat_shape,
                             Eigen::Tensor<float, 4> &output_eigen)
{
    const Eigen::Tensor<float, 4>::Dimensions & input_shape = input_eigen.dimensions();
    output_eigen = Eigen::Tensor<float, 4>(input_shape[0]*repeat_shape[0], input_shape[1]*repeat_shape[1],
                 input_shape[2]*repeat_shape[2], input_shape[3]*repeat_shape[3]);

    for(int l = 0; l < input_shape[0]*repeat_shape[0]; l++)
    {
        for(int k = 0; k < input_shape[1]*repeat_shape[1]; k++)
        {
            for(int j = 0; j < input_shape[2]*repeat_shape[2]; j++)
            {
                for(int i = 0; i < input_shape[3]*repeat_shape[3]; i++)
                {
                    output_eigen(l, k, j, i) = input_eigen(l%input_shape[0], k%input_shape[1],
                                                        j%input_shape[2], i%input_shape[3]);
                }
            }
        }
    }
}

void ClusterFunc::ScatterAddImpl(Eigen::Tensor<float, 4> &index_eigen,
                                 Eigen::Tensor<float, 4> &src_eigen,
                                 int add_dim,
                                 Eigen::Tensor<float, 4> &output_eigen)
{
    output_eigen.setZero();

    int batch_size = output_eigen.dimension(0);
    int channel = output_eigen.dimension(1);
    int height = output_eigen.dimension(2);
    int width = output_eigen.dimension(3);

    for(int i = 0; i < batch_size; i++)
    {
        for(int j = 0; j < channel; j++)
        {
            for(int k = 0; k < height; k++)
            {
                for(int l = 0; l < width; l++)
                {
                    int j_index = index_eigen(i,j,k,l);
                    output_eigen(i, j_index, k, l) += src_eigen(i, j, k, l);
                }
            }
        }
    }
}

// Eigen::Tensor<float, 3> ClusterFunc::reshape(Eigen::Tensor<int, 3> input, int channel, int height, int width)
// {
// 	Eigen::Tensor<float, 3> res;
// 	vector<float> one_dim_array;
// 	int index = 0;
// 	for (int i = 0; i < input.dimensions()[0]; i++)
// 	{
// 		for (int i2 = 0; i2 < input.dimensions()[1]; i2++)
// 		{
// 			for (int i3 = 0; i3 < input.dimensions()[2]; i3++)
// 			{
// 				one_dim_array.push_back(input(i, i2, i3));
// 			}
// 		}
// 	}
//  	res = Eigen::Tensor<float, 3>(Eigen::TensorMap<Eigen::Tensor<float, 3>>(one_dim_array.data(), channel, height, width));
// 	return res;
// }