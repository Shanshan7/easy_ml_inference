#include <iostream>
#include <map>

std::map<int, int> cluster_unique_map;

void ClusterUnique(int *cluster_labels, int cluster_labels_length)
{
    // int cluster_labels_length = sizeof(cluster_labels) / sizeof(int);
    std::cout << "cluster_labels_length: " << cluster_labels_length << std::endl;

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

int main()
{
    int cluster_labels[8] = {1,2,5,3,1,5,6,3};

	ClusterUnique(cluster_labels, 8);
	std::cout << cluster_unique_map.size() << std::endl;

	for (std::map<int, int>::iterator it = cluster_unique_map.begin(); it != cluster_unique_map.end(); ++it)
	{
		std::cout << "ele: " << it->first << " and counts is " << it->second << std::endl;
	}

    return 0;
}