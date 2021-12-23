#ifndef _DATA_STRUCT_H_
#define _DATA_STRUCT_H_


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <ctype.h>
#include <wchar.h>
#include <unistd.h>
#include <getopt.h>
#include <linux/fb.h>
#include <fcntl.h>
#include <dirent.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <vector>
#include <map>
#include <glog/logging.h>
#include <opencv2/core.hpp>

#define MOT_MAX_TRACK_NUM	        100


// track struct
// struct TrackAttribute {
// 	float fdet_conf;             // detect confidence
// 	int ncls;                    // class
// 	int ntrack_id;               // track id
// 	float fobject_loc[4];        // object location: Image Coordinate, x1, y1, x2, y2
// };

// struct TrackOutPut {
// 	TrackAttribute track_attri[MOT_MAX_TRACK_NUM];
// 	int nframe_index;
// 	int nvalid_track_count;
// };

/********************************************
行人方向：0 未知 1 正向 2 背向
********************************************/
enum PEDESTRIAN_DIRECTION
{
	E_PEDESTRIAN_DIRECTION_DONT_KNOWN = 0,
	E_PEDESTRIAN_DIRECTION_FORWARD = 1,
	E_PEDESTRIAN_DIRECTION_BACKWARD = 2
};

/********************************************
行人性别：0 未知 1 男性 2 女性
********************************************/
enum PEDESTRIAN_GENDER
{
	E_PEDESTRIAN_GENDER_DONT_KNOWN = 0,
	E_PEDESTRIAN_GENDER_MALE = 1,
	E_PEDESTRIAN_GENDER_FEMALE = 2
};

/********************************************
行人性别：0 未知 1 小于20 2 20到60 3 大于60
********************************************/
enum PEDESTRIAN_AGE
{
	E_PEDESTRIAN_AGE_DONT_KNOWN = 0,
	E_PEDESTRIAN_AGE_LESS_THAN_20 = 1,
	E_PEDESTRIAN_AGE_20_TO_60 = 2,
	E_PEDESTRIAN_AGE_MORE_THAN_60 = 3
};

/********************************************
头发类型：0 未知 1 短头发 2 长头发
********************************************/
enum HAIR_TYPE
{
	E_HAIR_TYPE_DONT_KNOWN = 0,
	E_HAIR_TYPE_SHORT = 1,
	E_HAIR_TYPR_LONG = 2
};

/********************************************
戴帽类型：0 未知 1 戴帽子 2 没带帽子
********************************************/
enum WEAR_CAP_TYPE
{
	E_WEAR_CAP_TYPE_DONT_KNOWN = 0,
	E_WEAR_CAP_TYPE_WEAR = 1,
	E_WEAR_CAP_TYPE_NO_WEAR = 2
};

/********************************************
上衣颜色：0 未知 1 上衣红色 2 上衣蓝色 3 上衣绿色
********************************************/
enum COAT_COLOR
{
	E_COAT_COLOR_DONT_KNOWN = 0,
	E_COAT_COLOR_RED = 1,
	E_COAT_COLOR_BLUE = 2,
	E_COAT_COLOR_GREEN = 3
};

/********************************************
上衣长短：0 未知 1 上衣长 2 上衣短
********************************************/
enum COAT_LENGTH
{
	E_COAT_LENGTH_DONT_KNOWN = 0,
	E_COAT_LENGTH_SHORT = 1,
	E_COAT_LENGTH_LONG = 2
};

/********************************************
袖子长短：0 未知 1 袖子长 2 袖子短
********************************************/
enum SLEEVE_LENGTH
{
	E_SLEEVE_LENGTH_DONT_KNOWN = 0,
	E_SLEEVE_LENGTH_SHORT = 1,
	E_SLEEVE_LENGTH_LONG = 2
};

/********************************************
下衣类型：0 未知 1 牛仔裤 2 裙子
********************************************/
enum LOWER_BODY_CLOTHES_TYPE
{
	E_LOWER_BODY_CLOTHES_TYPE_DONT_KNOWN = 0,
	E_LOWER_BODY_CLOTHES_TYPE_JEANS = 1,
	E_LOWER_BODY_CLOTHES_TYPE_SKIRT = 2
};

/********************************************
下衣长短：0 未知 1 下衣长 2 下衣短
********************************************/
enum LOWER_BODY_CLOTHES_LENGTH
{
	E_LOWER_BODY_CLOTHES_LENGTH_DONT_KNOWN = 0,
	E_LOWER_BODY_CLOTHES_LENGTH_SHORT = 1,
	E_LOWER_BODY_CLOTHES_LENGTH_LONG = 2
};

/********************************************
下衣颜色：0 未知 1 下衣红色 2 下衣蓝色 3 下衣绿色
********************************************/
enum LOWER_BODY_CLOTHES_COLOR
{
	E_LOWER_BODY_CLOTHES_COLOR_DONT_KNOWN = 0,
	E_LOWER_BODY_CLOTHES_COLOR_RED = 1,
	E_LOWER_BODY_CLOTHES_COLOR_BLUE = 2,
	E_LOWER_BODY_CLOTHES_COLOR_GREEN = 3
};

/********************************************
随身物品：0 未知 1 没有随身物品 2 背包 3 手提包 4 伞
********************************************/
enum PERSONAL_BELONGINGS
{
	E_PERSONAL_BELONGINGS_DONT_KNOWN = 0,
	E_PERSONAL_BELONGINGS_NOTHING = 1,
	E_PERSONAL_BELONGINGS_BAG = 2,
	E_PERSONAL_BELONGINGS_HANDBAG = 3,
	E_PERSONAL_BELONGINGS_UMBRELLA = 4
};

/********************************************
鞋子状态：0 未知 1 鞋子亮 2 鞋子暗
********************************************/
enum SHOES_STATUS
{
	E_SHOES_STATUS_DONT_KNOWN = 0,
	E_SHOES_STATUS_SHOES_BRIGHT = 1,
	E_SHOES_STATUS_SHOES_DARK = 2
};

/********************************************
靴子类型：0 未知 1 穿靴子 2 没穿靴子
********************************************/
enum BOOTS_TYPE
{
	E_BOOTS_TYPE_DONT_KNOWN = 0,
	E_BOOTS_TYPE_BOOTS = 1,
	E_BOOTS_TYPE_NO_BOOTS = 2
};

struct YUVDataInfo
{
	int data_width;
	int data_height;
	long phy_address[3];
	char *vir_address[3];
	int data_stride[3];
};


struct PedestrianParams
{
	// pedestria param
    PEDESTRIAN_DIRECTION pedestrian_direction;
	PEDESTRIAN_AGE pedestrian_age;
	PEDESTRIAN_GENDER pedestrian_gender;

	HAIR_TYPE hair_style;
	WEAR_CAP_TYPE wear_cap;
	COAT_COLOR coat_color;
	COAT_LENGTH coat_length;
	SLEEVE_LENGTH sleeve_length;
	LOWER_BODY_CLOTHES_TYPE lower_body_clothes_type;
	LOWER_BODY_CLOTHES_LENGTH lower_body_clothes_length;
	LOWER_BODY_CLOTHES_COLOR lower_body_clothes_color;
	PERSONAL_BELONGINGS personal_belongings;
	SHOES_STATUS shoes_status;
	BOOTS_TYPE boots_type;
};


struct DetectResult {
    float head_location[4];                      // the location of head [x1, y1, x2, y2]
	float pedestrian_location[4];                // the location of pedestrian [x1, y1, x2, y2]
    float confidence;
    float class_id;
    float track_id;

	PedestrianParams pedestrian_params;
};

struct DetectResultInfo {
	int current_frame;
	YUVDataInfo yuv_data;
	std::vector<DetectResult> detect_result_vector;
};

struct TrajectoryParams {
	int latest_frame_id=-1;                         // the latest frame the target has captured recently
	int draw_flag;

	// Pedestrian param
	PedestrianParams pedestrian_params;

	// Track param
	int num_pedestrian;
	int pedestrian_direction;
	float relative_distance;
	float head_location[4];
	float mean_velocity;
	std::vector<float> velocity_vector;
	// std::vector<float> pedestrian_x_start;
	// std::vector<float> pedestrian_y_start;
	// std::vector<float> pedestrian_x_end;
	// std::vector<float> pedestrian_y_end;
	float pedestrian_location[4];
	std::vector<cv::Point2f> trajectory_position;
	std::vector<cv::Point2f> trajectory_bird_position;
};

// struct GlobalControlParam {
// 	int run_flag;
// 	int current_frame;
// 	std::string camera_calibration_file;
	
// 	// results
// 	DetectResult detect_result;
// 	std::map<int, TrajectoryParams> track_idx_map;
// };


// /********************************************
// Function：calculate trajectory
// Description:  Track the target and output speed, direction and position
// Input:	det_result_info
// OutPut: track_idx_map
// ********************************************/

// class CalculateTraj{
// public:
//     CalculateTraj();
//     ~CalculateTraj();
// 	void calculate_trajectory(DetectResultInfo &det_result_info);
// 	void save_det_result(DetectResultInfo &det_result_info);               
// 	// save detect result like: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <0/1忽略>, <cls>, <>

// public:
// 	std::map<int, TrajectoryParams> track_idx_map;

// private:
//     void bird_view_transform(DetectResultInfo &det_result_info);

// private:
// 	int run_flag;
// 	int dTs;
// 	float pixel2world_distance;
// 	std::string save_txt_path;
// 	std::string camera_calibration_file;
// };

#endif // _DATA_STRUCT_H_
