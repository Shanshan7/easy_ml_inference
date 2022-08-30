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
#include <unordered_map>
#include <dirent.h>
#include <sys/types.h>    
#include <sys/stat.h> 
#include <sys/time.h>

//Eigen
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

//Boost
#include <boost/tokenizer.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

// PCL
// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

//opencv
#include <opencv2/imgcodecs.hpp>

#include "lonlat2utm.h"
#include "readparam.h"
#include "tracker.h"
#include "smoke.h"
#include "utils.h"

using namespace std;
typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
static string folder_lidar = "results/";
static string format_lidar = "%|06|.txt";

struct pose{
	double x;
	double y;
	double heading;
};

template <class Type>  
Type stringToNum(const string& str)  
{  
    istringstream iss(str);  
    Type num;  
    iss >> num;  
    return num;      
}  

template<typename T> string toString(const T& t) {
	ostringstream oss;
	oss << t;
	return oss.str();
}

int fileNameFilter(const struct dirent *cur) {
	std::string str(cur->d_name);
	if (str.find(".bin") != std::string::npos
			|| str.find(".pcd") != std::string::npos
			|| str.find(".png") != std::string::npos
			|| str.find(".jpg") != std::string::npos
			|| str.find(".txt") != std::string::npos) {
		return 1;
	}
	return 0;
}

// with open(os.path.join(self.calib_dir, file_name), 'r') as csv_file:
// 	reader = csv.reader(csv_file, delimiter=' ')
// 	for line, row in enumerate(reader):
// 		if row[0] == 'P2:':
// 			K = row[1:]
// 			K = [float(i) for i in K]
// 			K = np.array(K, dtype=np.float32).reshape(3, 4)
// 			K = K[:3, :3]
// 			break

// return annotations, 


bool get_all_files(const std::string& dir_in,
		std::vector<std::string>& files) {

	if (dir_in.empty()) {
		return false;
	}
	struct stat s;
	stat(dir_in.c_str(), &s);
	if (!S_ISDIR(s.st_mode)) {
		return false;
	}
	DIR* open_dir = opendir(dir_in.c_str());
	if (NULL == open_dir) {
		std::exit(EXIT_FAILURE);
	}
	dirent* p = nullptr;
	while ((p = readdir(open_dir)) != nullptr) {
		struct stat st;
		if (p->d_name[0] != '.') {
			//因为是使用devC++ 获取windows下的文件，所以使用了 "\" ,linux下要换成"/"
			//cout<<std::string(p->d_name)<<endl;
			std::string name = dir_in + std::string("/")
					+ std::string(p->d_name);
			stat(name.c_str(), &st);
			if (S_ISDIR(st.st_mode)) {
				get_all_files(name, files);
			} else if (S_ISREG(st.st_mode)) {
				boost::char_separator<char> sepp { "." };
				tokenizer tokn(std::string(p->d_name), sepp);
				vector<string> filename_sep(tokn.begin(), tokn.end());
				string type_ = "." + filename_sep[1];
				break;
			}
		}
	}

	struct dirent **namelist;
	int n = scandir(dir_in.c_str(), &namelist, fileNameFilter, alphasort);
	if (n < 0) {
		return false;
	}
	for (int i = 0; i < n; ++i) {
		std::string filePath(namelist[i]->d_name);
		files.push_back(filePath);
		free(namelist[i]);
	};
	free(namelist);
	closedir(open_dir);
	return true;
}

bool Load_Sensor_Data_Path(std::vector<std::string>& lidarfile_name, std::vector<std::string>& imagefile_name, string& path){
	// string lidar_file_path = path + "/velodyne";
	// cout<<lidar_file_path<<endl;
	// if(!get_all_files(lidar_file_path, lidarfile_name))
	// 	return false;
	string image_file_path = path + "/image_02";
	cout<<image_file_path<<endl;
	if(!get_all_files(image_file_path, imagefile_name))
		return false;
	// if(lidarfile_name.size()!= imagefile_name.size())
	// 	return false;
	return true;
}

void draw3dbox(DetectStruct &det, cv::Mat& image, vector<int>& color, int id){
	float h = det.box[0];	
	float w = det.box[1];
	float l = det.box[2];
    float x = det.position[0];
	float y = det.position[1];
	float z = det.z;
	float yaw = -det.yaw - 90 * M_PI / 180;
	double boxroation[9] = {cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1 };

	Eigen::MatrixXd BoxRotation = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> >(boxroation);
	double xAxisP[8] = {l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2 }; 
	double yAxisP[8] = {w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2}; 
#ifdef USE_SMOKE
	double zAxisP[8] = {-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2}; 
#else
	double zAxisP[8] = {h, h, h, h, 0, 0, 0, 0}; 
#endif
	vector<cv::Point> imagepoint;
	Eigen::Vector3d translation(x,y,z);

	for (int i = 0; i < 8; i++) {
		Eigen::Vector3d point_3d(xAxisP[i], yAxisP[i], zAxisP[i]);
		Eigen::Vector3d rotationPoint_3d = BoxRotation * point_3d + translation;
		cv::Point imgpoint = cloud2camera(rotationPoint_3d);
		imagepoint.push_back(imgpoint);
	}

	int r = color[0];
	int g = color[1];
	int b = color[2];

	// tmp filter method(利用地面模型)
#ifdef USE_SMOKE
	float image_ground_ex = image.rows / 2;
#else
	float image_ground_ex = 0;
#endif
	float image_corner = imagepoint[0].y;
	if (image_corner > image_ground_ex)
	{
		cv::line(image, imagepoint[0], imagepoint[1],cv::Scalar(226, 43, 138), 2, CV_AA);
		cv::line(image, imagepoint[1], imagepoint[2],cv::Scalar(r, g, b), 2, CV_AA);
		cv::line(image, imagepoint[2], imagepoint[3],cv::Scalar(r, g, b), 2, CV_AA);
		cv::line(image, imagepoint[3], imagepoint[0],cv::Scalar(r, g, b), 2, CV_AA);

		cv::line(image, imagepoint[4], imagepoint[5],cv::Scalar(226, 43, 138), 2, CV_AA);
		cv::line(image, imagepoint[5], imagepoint[6],cv::Scalar(r, g, b), 2, CV_AA);
		cv::line(image, imagepoint[6], imagepoint[7],cv::Scalar(r, g, b), 2, CV_AA);
		cv::line(image, imagepoint[7], imagepoint[4],cv::Scalar(r, g, b), 2, CV_AA);

		cv::line(image, imagepoint[0], imagepoint[4],cv::Scalar(226, 43, 138), 2, CV_AA);
		cv::line(image, imagepoint[1], imagepoint[5],cv::Scalar(226, 43, 138), 2, CV_AA);
		cv::line(image, imagepoint[2], imagepoint[6],cv::Scalar(r, g, b), 2, CV_AA);
		cv::line(image, imagepoint[3], imagepoint[7],cv::Scalar(r, g, b), 2, CV_AA);

		cv::line(image, imagepoint[0], imagepoint[5],cv::Scalar(226, 43, 138), 2, CV_AA);
		cv::line(image, imagepoint[1], imagepoint[4],cv::Scalar(226, 43, 138), 2, CV_AA);

		std::string lbl = cv::format("ID: %d", id);
		cv::putText(image, lbl, (imagepoint[0], imagepoint[1]), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(r, g, b));

		imagepoint.clear();
	}
}

int main(int argc, char** argv){
	std::string datapath = "/data/kitti_mini";
	std::string file = "0020";
	std::string trackclass = "Car";

#ifdef USE_SMOKE
	cv::Mat intrinsic = (cv::Mat_<float>(3, 3) << 
			721.5377, 0.0, 609.5593, 0.0, 721.5377, 172.854, 0.0, 0.0, 1.0);
	SMOKE smoke("../data/models/smoke_dla34.trt8", intrinsic);
#endif

	Param param;
	Tracker tracker(param);
	std::string path = datapath+file;
	cout<<path<<endl;
	std::vector<std::string>  lidarname;
	std::vector<std::string>  imagename;
	float time = 0.1;

	if(!Load_Sensor_Data_Path(lidarname, imagename, path)){
		cout<<"Detecion file wrong!"<<endl;
		std::abort();
	}

	std::unordered_map<string, int> classname2id;
	classname2id["Car"] = 1;
	classname2id["Pedestrian"] = 2;	
	classname2id["Cyclist"] = 3;	

	//read gps data get the longitude latitude
	std::string gpspath = path + "/oxts/"+file+".txt";
	cout<<gpspath<<endl;
	std::ifstream gps(gpspath);
	std::vector<std::string> gpsdata;
	if (gps) {
		boost::char_separator<char> sep_line { "\n" };
		std::stringstream buffer;
		buffer << gps.rdbuf();
		std::string contents(buffer.str());
		tokenizer tok_line(contents, sep_line);
		std::vector<std::string> lines(tok_line.begin(), tok_line.end());
		gpsdata = lines;
		int size = gpsdata.size();
	}

	boost::char_separator<char> sep { " " };
	int maxframe = 0;
	unordered_map<int, vector<DetectStruct>> Inputdets;

#ifdef USE_DET_FILES
	// read the label file
	std::string labelpath = path + "/label_02/"+file+".txt";
	cout<<labelpath<<endl;
	std::ifstream label(labelpath);
	std::vector<std::string> labeldata;
	if (label) {
		boost::char_separator<char> sep_line { "\n" };
		std::stringstream buffer;
		buffer << label.rdbuf();
		std::string contents(buffer.str());
		tokenizer tok_line(contents, sep_line);
		std::vector<std::string> lines(tok_line.begin(), tok_line.end());
		labeldata = lines;
		int size = labeldata.size();
		
		tokenizer tokn(labeldata[size-1], sep);
		vector<string> temp_sep(tokn.begin(), tokn.end());
		maxframe = stringToNum<int>(temp_sep[0]);
		cout<<maxframe<<" "<< lidarname.size()<<endl;
	}

	int max_truncation = 0;
	int max_occlusion = 2;

	for(int i=0; i < labeldata.size(); ++i){
		tokenizer tokn(labeldata[i], sep);
		vector<string> temp_sep(tokn.begin(), tokn.end());
		int fra        =  stringToNum<int>(temp_sep[0]);    //frame
		string type    = temp_sep[2];                       //class
		int truncation = stringToNum<int>(temp_sep[3]);     //truncation
		int occlusion  = stringToNum<int>(temp_sep[4]);     //occlusion
		float x1       = stringToNum<float>(temp_sep[6]);   //left[pixel]
		float y1       = stringToNum<float>(temp_sep[7]);   //top[pixel]
		float x2       = stringToNum<float>(temp_sep[8]);   //right[pixel]
		float y2       = stringToNum<float>(temp_sep[9]);   //bottom[pixel]
		float h        = stringToNum<float>(temp_sep[10]);  //h[m]
		float w        = stringToNum<float>(temp_sep[11]);  //w[m]
		float l        = stringToNum<float>(temp_sep[12]);  //l[m]
		float x        = stringToNum<float>(temp_sep[13]);  //x[m]
		float y        = stringToNum<float>(temp_sep[14]);  //y[m]
		float z        = stringToNum<float>(temp_sep[15]);  //z[m]
		float yaw      = stringToNum<float>(temp_sep[16]);  //yaw angle[rad]

		if(truncation>max_truncation || occlusion>max_occlusion || classname2id[type] != classname2id[trackclass])
			continue;

		Eigen::Vector3d cpoint;
		cpoint<<x,y,z;
		//cout<<"came " <<cpoint<<"\n"<<endl;

		Eigen::Vector3d ppoint = camera2cloud(cpoint);
		//cout<<"cloud: "<< ppoint<<"\n"<<endl;
		
		DetectStruct det;
		det.box2D.resize(4);
		det.box.resize(3);
		det.classname  = classname2id[type];
		det.box2D[0]   = x1;
		det.box2D[1]   = y1;
		det.box2D[2]   = x2;
		det.box2D[3]   = y2;
		det.box[0]     = h;//h
		det.box[1]     = w;//w
		det.box[2]     = l;//l
		det.z          = ppoint(2);
		det.yaw        = yaw;
		det.position   = Eigen::VectorXd(2);
		det.position << ppoint(0),ppoint(1);

		Inputdets[fra].push_back(det);
	}
#endif

	// start position
	Eigen::Matrix3d rotZorigion; 
	Eigen::Isometry3d porigion;
	double oriheading=0;
	double orix =0;
	double oriy = 0;

	Eigen::Matrix3d rotZpre;
	Eigen::Isometry3d ppre;
	double preheading=0;
	double prex =0;
	double prey = 0;
	cv::Mat images = cv::Mat::zeros(608, 608, CV_8UC3);

	double totaltime = 0;

	int frame = 0;

	cv::RNG rng(12345);
	unordered_map<int, vector<int>> idcolor;

	/*cv::VideoWriter writer;

	writer.open(path+"tracking.avi",CV_FOURCC('M', 'J', 'P', 'G'), 10 , cv::Size(1241, 376), true);
    	if (!writer.isOpened())
    	{
        	std::abort();
    	}*/

	std::ofstream file_p;
	while(frame < 10000000){
		//get the gps data and get the tranformation matrix
		tokenizer tokn(gpsdata[frame], sep);
		vector<string> temp_sep(tokn.begin(), tokn.end());

		double UTME,UTMN;
		double latitude = stringToNum<double>(temp_sep[0]);
		double longitude = stringToNum<double>(temp_sep[1]);
		double heading = stringToNum<double>(temp_sep[5]) - 90 * M_PI/180;
		LonLat2UTM(longitude, latitude, UTME, UTMN);

		Eigen::Isometry3d translate2origion;
		Eigen::Isometry3d origion2translate;

		double twosub = 0;

		//read image
		string impath = path + "/image_02/" + imagename[frame];
		cv::Mat rgbimage = cv::imread(impath);
#ifdef USE_SMOKE
		smoke.Detect(rgbimage);
		smoke.PostProcess(rgbimage, Inputdets[frame]);
#endif

		if(frame == 0){
			Eigen::AngleAxisd roto(heading ,Eigen::Vector3d::UnitZ());
			rotZorigion = roto.toRotationMatrix();
			porigion = rotZorigion;
			orix = UTME;
			oriy = UTMN;
			porigion.translation() = Eigen::Vector3d(UTME, UTMN, 0);
			oriheading = heading;

			preheading = heading;
			prex = UTME;
			prey = UTMN;
			rotZpre = rotZorigion;
			ppre = rotZpre;
			ppre.translation() = Eigen::Vector3d(UTME, UTMN, 0);
		}
		else{
			Eigen::AngleAxisd rotnow(heading , Eigen::Vector3d::UnitZ());
			Eigen::Matrix3d rotpnow = rotnow.toRotationMatrix();
			Eigen::Isometry3d p2;
			p2 = rotpnow;
			p2.translation() = Eigen::Vector3d(UTME, UTMN, 0);
			translate2origion = porigion.inverse() * p2;
			origion2translate = p2.inverse() * porigion;
			twosub = heading - oriheading;
		}

		int size = Inputdets[frame].size();
		// cout<<"inpusize "<<size<<endl;

		for(int i = 0; i < size; ++i){
			// file_p.open("./result.txt", ios::app);

			// // vector<int> color = {0, 255, 0};
			// // draw3dbox(Inputdets[frame][i], rgbimage, color, 0);

			// // cout<<"input: "<<Inputdets[frame][i].position(0)<<" "<<Inputdets[frame][i].position(1)<<" "<<Inputdets[frame][i].z<<" "<<Inputdets[frame][i].box[0]<<" "<<Inputdets[frame][i].box[1]<<" "<<Inputdets[frame][i].box[2]<<endl;
			// file_p << frame << " " << -1 << " " << Inputdets[frame][i].classname << " " << 0 << " " << 0 << " " \
			//        << -10 << " " << 10 << " " << 100 << " " << 10 << " " << 100 << " " << Inputdets[frame][i].box[0] << " "  \
			// 	   << Inputdets[frame][i].box[1] << " " << Inputdets[frame][i].box[2] << " " \
			// 	   << Inputdets[frame][i].position(0) << " " << Inputdets[frame][i].position(0) << " " \
			// 	   << Inputdets[frame][i].z << " " << Inputdets[frame][i].yaw << "\n";
			// file_p.close();

			Eigen::VectorXd v(2,1);
			v(1)   = Inputdets[frame][i].position(0);//x in kitti lidar
			v(0)   = -Inputdets[frame][i].position(1);//y in kitti lidar
			if(frame!=0){
				Eigen::Vector3d p_0(v(0), v(1), 0);
				Eigen::Vector3d p_1;
				p_1 = translate2origion * p_0;
				v(0) = p_1(0);
				v(1) = p_1(1);
			}

			Inputdets[frame][i].position(0) = v(0);
			Inputdets[frame][i].position(1) = v(1);

			Inputdets[frame][i].rotbox = cv::RotatedRect(cv::Point2f((v(0)+25)*608/50, v(1)*608/50), 
							cv::Size2f(Inputdets[frame][i].box[1]*608/50, Inputdets[frame][i].box[2]*608/50), Inputdets[frame][i].yaw);				     

			cv::RotatedRect detshow = cv::RotatedRect(cv::Point2f((v(0)+25)*608/50, v(1)*608/50), 
							cv::Size2f(Inputdets[frame][i].box[1]*608/50, Inputdets[frame][i].box[2]*608/50), Inputdets[frame][i].yaw);
			cv::Point2f vertices[4];
			detshow.points(vertices);
	    	for (int j = 0; j < 4; j++)
			{
				cv::line(images, vertices[j], vertices[(j+1)%4], cv::Scalar(0,0,255), 1);
			}
		}

		std::vector<Eigen::VectorXd> result;
		int64_t tm0 = gtm();
		tracker.track(Inputdets[frame], time, result);
   		int64_t tm1 = gtm();
  		printf("[INFO]update cast time:%ld us\n",  tm1-tm0);

		double x = tm1-tm0;
		totaltime += x;

		int marker_id = 0;
		for(int i=0; i<result.size(); ++i){
			Eigen::VectorXd r = result[i];
			if(frame != 0){
				Eigen::Vector3d p_0(r(1), r(2), 0);
				Eigen::Vector3d p_1;
				p_1 = origion2translate * p_0;
				r(1) = p_1(0);
				r(2) = p_1(1);
			}

			DetectStruct det;
			det.box2D.resize(4);
			det.box.resize(3);
			det.box[0]     = r(9);//h
			det.box[1]     = r(8);//w
			det.box[2]     = r(7);//l
			det.z          = r(10);
			det.yaw        = r(6);
			det.position   = Eigen::VectorXd(2);
			det.position << r(2), -r(1);
			// cout<<"det: "<<det.position(0)<<" "<<det.position(1)<<" "<<det.z<<" "<<det.box[0]<<" "<<det.box[1]<<" "<<det.box[2]<<endl;

			if (!idcolor.count(int(r(0)))){
                		int red = rng.uniform(0, 255);
                		int green = rng.uniform(0, 255);
                		int blue = rng.uniform(0, 255);			
				idcolor[int(r(0))] = {red,green,blue};
			}
			draw3dbox(det, rgbimage, idcolor[int(r(0))], int(r(0)));
		}
		cv::imshow("x", rgbimage);
		cv::waitKey(1);
		//writer<<rgbimage;

		time+=0.1;
		frame++;
	}
	//writer.release();

	return 0;
}