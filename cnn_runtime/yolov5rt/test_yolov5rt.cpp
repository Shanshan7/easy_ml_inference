#include <yolov5.h>
#include <deepsort.h>
#include <utils.h>
#include <glog/logging.h>


static void showDetection(cv::Mat& img, std::vector<DetectBox>& boxes) {
    cv::Mat temp = img.clone();
    for (auto box : boxes) {
        cv::Point lt(box.x1, box.y1);
        cv::Point br(box.x2, box.y2);
        cv::rectangle(temp, lt, br, cv::Scalar(255, 0, 0), 1);
        //std::string lbl = cv::format("ID:%d_C:%d_CONF:%.2f", (int)box.trackID, (int)box.classID, box.confidence);
		//std::string lbl = cv::format("ID:%d_C:%d", (int)box.trackID, (int)box.classID);
		std::string lbl = cv::format("ID:%d_x:%.1f_y:%.1f",(int)box.trackID,(box.x1+box.x2)/2,(box.y1+box.y2)/2);
        cv::putText(temp, lbl, lt, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,255,0));
    }
    cv::imshow("img", temp);
    cv::waitKey(1);
}

int main(int argc, char** argv)
{
    int rval = 0;
    google::InitGoogleLogging(argv[0]);
    google::SetLogDestination(google::INFO, "../log/");
    // google::InstallFailureSignalHandler();
	// google::InstallFailureWriter(&SignalHandle); 

    string yolo_path = "../yolov5/yolov5s.onnx";
    string sort_path = "../deepsort/deepsort.onnx";

    LOG(INFO) << "yolo_path: " << yolo_path;

    if(argc != 2)
	{
        printf("usage: ./test_yolov5rt image_dir\n");
        exit(0);
    }
	std::vector<std::string> images;
	ListImages(argv[1], images);
    std::cout << "total Test images : " << images.size() << std::endl;

    YOLO yolo_model(yolo_path);
    DeepSort* DS = new DeepSort(sort_path, 128, 256, 0);

    // yolo detect
    std::vector<DetectBox> det_results;
	for (int index = 0; index < images.size(); index++) 
	{
		std::stringstream temp_str;
     	temp_str << argv[1] << index+1 << ".jpg";
		std::cout << temp_str.str() << std::endl;
		cv::Mat frame = cv::imread(temp_str.str());

        det_results.clear();

        unsigned long time_start_yolo, time_end_yolo, time_start_sort, time_end_sort;
        time_start_yolo = get_current_time();
        yolo_model.run(frame, det_results);
        time_end_yolo = get_current_time();
        std::cout << "yolov5 cost time: " <<  (time_end_yolo - time_start_yolo)/1000.0  << "ms" << std::endl;
        // std::cout << "det: " << det_results.size() << std::endl;
        time_start_sort = get_current_time();
        DS->sort(frame, det_results);
        time_end_sort = get_current_time();
        std::cout << "deepsort cost time: " <<  (time_end_sort - time_start_sort)/1000.0  << "ms" << std::endl;
        showDetection(frame, det_results);
    }

    google::ShutdownGoogleLogging();
    return rval;
}