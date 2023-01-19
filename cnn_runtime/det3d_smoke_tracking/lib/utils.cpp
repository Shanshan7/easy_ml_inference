#include "utils.h"

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

int64_t gtm() {
	struct timeval tm;
	gettimeofday(&tm, 0);
	// return ms
	int64_t re = (((int64_t) tm.tv_sec) * 1000 * 1000 + tm.tv_usec);
	return re;
}

Eigen::Vector3d camera2cloud(Eigen::Vector3d input){
	Eigen::Matrix4d RT_velo_to_cam;
	Eigen::Matrix4d R_rect;
    	RT_velo_to_cam<<7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03,
    			1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02,
    			9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01,
			0, 0, 0, 1;
    	R_rect<<0.99992475, 0.00975976, -0.00734152, 0,
    		-0.0097913, 0.99994262, -0.00430371, 0,
    		0.00729911, 0.0043753, 0.99996319, 0,
   		0, 0, 0, 1;

	Eigen::Vector4d point;
	point<<input(0), input(1), input(2), 1;
	Eigen::Vector4d pcloud = RT_velo_to_cam.inverse()*point; // R_rect.inverse()* 
	Eigen::Vector3d result;
	result<<pcloud(0),pcloud(1),pcloud(2);
	return result;
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

cv::Point cloud2camera(Eigen::Vector3d input){
	Eigen::Matrix4d RT_velo_to_cam;
	Eigen::Matrix4d R_rect;
	Eigen::MatrixXd project_matrix(3,4);
    	RT_velo_to_cam<<7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03,
    			1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02,
    			9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01,
			0, 0, 0, 1;
    	// R_rect<<0.99992475, 0.00975976, -0.00734152, 0,
    	// 	-0.0097913, 0.99994262, -0.00430371, 0,
    	// 	0.00729911, 0.0043753, 0.99996319, 0,
   		// 0, 0, 0, 1;
    	// project_matrix<<7.215377e+02,0.000000e+00,6.095593e+02,4.485728e+01,
        //             0.000000e+00,7.215377e+02,1.728540e+02,2.163791e-01,
        //             0.000000e+00,0.000000e+00,1.000000e+00,2.745884e-03;
		project_matrix<< 1015.2, 0.0, 960.2334, -0.3246,
										0.0, 1015.5, 487.1393, 0.1029,
										0.0, 0.0, 1.0, -0.0151;
    	Eigen::MatrixXd transform_matrix_ = project_matrix*RT_velo_to_cam; //R_rect

	Eigen::Vector4d point;
	point<<input(0), input(1), input(2), 1;
	Eigen::Vector3d pimage = transform_matrix_* point;
	cv::Point p2d = cv::Point(pimage(0)/pimage(2),pimage(1)/pimage(2));
	return p2d;
}

float Sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}