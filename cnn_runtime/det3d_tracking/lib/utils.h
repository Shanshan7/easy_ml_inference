#include <iostream>
#include <vector>
#include <stdlib.h>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>
#include <time.h> 
#include <sys/types.h>    
#include <sys/stat.h> 
#include <sys/time.h>

//Eigen
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

//opencv
#include <opencv2/imgcodecs.hpp>


int64_t gtm();
Eigen::Vector3d camera2cloud(Eigen::Vector3d input);
cv::Point cloud2camera(Eigen::Vector3d input);
float Sigmoid(float x);