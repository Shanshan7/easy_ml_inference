#include <iostream>
#include <fstream>
#include <math.h>
//opencv
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include "patchcore_postprocess.h"

#include <sys/time.h>

#define KNEIGHBOURS (9)


int main()
{
    std::ifstream embedding("./embedding_train.bin", std::ios::binary|std::ios::in);
    embedding.seekg(0,std::ios::end);
    int embedding_length = embedding.tellg() / sizeof(float);
    embedding.seekg(0, std::ios::beg);
    float* embedding_coreset = new float[embedding_length];
    embedding.read(reinterpret_cast<char*>(embedding_coreset), sizeof(float) * embedding_length);
    embedding.close();
    cv::Mat embedding_train(194, 1440, CV_32FC1);

    memcpy(embedding_train.data, embedding_coreset, embedding_length * sizeof(float));

    std::ifstream embedding_test_file("./embedding_test.bin", std::ios::binary|std::ios::in);
    embedding_test_file.seekg(0,std::ios::end);
    int embedding_test_length = embedding_test_file.tellg() / sizeof(float);
    embedding_test_file.seekg(0, std::ios::beg);
    float* embedding_t = new float[embedding_test_length];
    embedding_test_file.read(reinterpret_cast<char*>(embedding_t), sizeof(float) * embedding_test_length);
    embedding_test_file.close();
    cv::Mat embedding_test(324, 1440, CV_32FC1);

    memcpy(embedding_test.data, embedding_t, embedding_test_length * sizeof(float));

    // cv::Mat embedding_train(194, 1440, CV_32FC1);
    // for (int a = 0; a < 194; a++)
    // {
    //     for (int b = 0; b < 1440; b++)
    //     {
    //         // float min_s = (float)a/19.0;
    //         // float ad_s = (float)b/40.0;
    //         embedding_train.at<float>(a,b) = 1.5; // 1.5 - min_s + ad_s;
    //     }
    // }


    // cv::Mat embedding_test(324, 1440, CV_32FC1);
    // for (int a = 0; a < 324; a++)
    // {
    //     for (int b = 0; b < 1440; b++)
    //     {
    //         // float add_s = (float)a/32.0;
    //         // float mi_s = (float)b/40.0;
    //         embedding_test.at<float>(a,b) = 1.2; // 1.2 + add_s - mi_s;
    //     }
    // }

    int ef_construction = KNEIGHBOURS * 5;
    int M = 16;
    // Declaring index，声明索引类型，如：l2, cosine or ip
    auto space = std::unique_ptr<hnswlib::L2Space>(new hnswlib::L2Space(embedding_train.cols));
    // 初始化索引，元素的最大数需要是已知的
    auto appr_alg = std::unique_ptr<hnswlib::HierarchicalNSW<float>>(
            new hnswlib::HierarchicalNSW<float>(space.get(), embedding_train.rows, M,
                                                ef_construction));

    int vecdim = embedding_train.cols;
    for(int t = 0; t < embedding_train.rows; t++)
    {
        float *mass_train = new float[vecdim];
        // memcpy(mass_train, embedding_train.data + t*vecdim, vecdim * sizeof(float));
		for (size_t j = 0; j < vecdim; j++)
		{
			mass_train[j] = embedding_train.at<float>(t, j);
		}
        appr_alg->addPoint((void*)mass_train, t);
        delete[] mass_train;
    }

    // process
    appr_alg->setEf(ef_construction);
    float *distances_mat = new float[embedding_test.rows*KNEIGHBOURS];
    float *distances = new float[embedding_test.rows*KNEIGHBOURS];
	for(int l = 0; l < embedding_test.rows; l++)
	{
		float *mass_test = new float[vecdim];
        // memcpy(mass_test, embedding_test.data + l*vecdim, vecdim * sizeof(float));
		for (int j = 0; j < vecdim; j++)
		{
			mass_test[j] = embedding_test.at<float>(l, j);
		}
		std::priority_queue<std::pair<float, hnswlib::labeltype>> candidates = 
                                                        appr_alg->searchKnn((void*)mass_test, KNEIGHBOURS);
		delete[] mass_test;

        for (int k = 0; k < KNEIGHBOURS; k++)
        {
            // std::cout << "l: " << l << ", k: " << k << ", dist: " << candidates.top().first << std::endl;
            // distances[k*embedding_test.rows + l] = candidates.top().first;
            // distances_mat[l*KNEIGHBOURS + (KNEIGHBOURS-k-1)] = candidates.top().first;
            distances[(KNEIGHBOURS-k-1)*embedding_test.rows + l] = candidates.top().first;
            candidates.pop();
        }
        // for (int m = 0; m < KNEIGHBOURS; m++)
        // {
        //     std::cout << "l: " << l << ", k: " << m << ", dist: " << distances[KNEIGHBOURS-m-1] << std::endl;
        // }
        // delete[] distances;
	}
    // reshape 784 * 9 --> 9 * 784
    // for (int d = 0; d < KNEIGHBOURS; d++)
    // {
    //     for (int c = 0; c < embedding_test.rows; c++)
    //     {
    //         distances[d*embedding_test.rows + c] = distances_mat[c*KNEIGHBOURS + d];
    //         // memcpy(distances + d*784 + c, distances_mat.data + c*9 + d, sizeof(float));
    //     }
    // }

    int max_posit = std::max_element(distances, \
				                distances + embedding_test.rows) - distances; // - distances.data;
    float* N_b = new float[KNEIGHBOURS];
    for(int i = 0; i < KNEIGHBOURS; i++)
    {
        N_b[i] = distances[i * embedding_test.rows + max_posit];
    }

    float w, sum_N_b = 0;
    for(int j = 0; j < KNEIGHBOURS; j++)
    {
        sum_N_b += exp(N_b[j]);
    }
    float max_N_b = *std::max_element(N_b, N_b + KNEIGHBOURS);
    w = (1 - exp(max_N_b) / sum_N_b);

    float score;
    score = w * distances[max_posit];
    std::cout << score << std::endl;
    
    return 0;
}