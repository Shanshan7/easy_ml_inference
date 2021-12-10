#include "yolo.h"
#include "../../utility/utils.h"

YOLO::YOLO(Net_config config)
{
	cout << "Net use " << config.netname << endl;
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
	strcpy(this->netname, config.netname.c_str());

	ifstream ifs(this->classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->classes.push_back(line);

	string modelFile = this->netname;
	modelFile += ".onnx";
	this->net = readNet(modelFile);
}

void YOLO::drawPred(Mat& frame, std::vector<DetResults> &det_results)   
// Draw the predicted bounding box
{
	for (int i = 0; i < det_results.size(); i++) {
		int class_idx = det_results[i].class_id;
		float conf = det_results[i].confidence;
		Rect box = det_results[i].box;
		int left = det_results[i].box.x;
		int top = det_results[i].box.y;
		int right = det_results[i].box.x + det_results[i].box.width;
		int bottom = det_results[i].box.y + det_results[i].box.height;
		//Draw a rectangle displaying the bounding box
		rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);

		//Get the label for the class name and its confidence
		string label = format("%.2f", conf);
		label = this->classes[class_idx] + ":" + label;

		//Display the label at the top of the bounding box
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
	}
}

void YOLO::save_txt(ofstream &labelfile, std::vector<DetResults> &det_results)
{
	for (int i = 0; i < det_results.size(); i++) {
		int class_idx = det_results[i].class_id;
		float conf = det_results[i].confidence;
		Rect box = det_results[i].box;
		int left = det_results[i].box.x;
		int top = det_results[i].box.y;
		int right = det_results[i].box.x + det_results[i].box.width;
		int bottom = det_results[i].box.y + det_results[i].box.height;

		labelfile << class_idx << " " << left \
				<< " " << top << " " << right \
				<< " " << bottom << " " << conf << "\n";
	}
}

void YOLO::sigmoid(Mat* out, int length)
{
	float* pdata = (float*)(out->data);
	int i = 0; 
	for (i = 0; i < length; i++)
	{
		pdata[i] = 1.0 / (1 + expf(-pdata[i]));
	}
}

void YOLO::run(Mat& frame, std::vector<DetResults> &det_results)
{
	Mat blob;
	int col = frame.cols;
	int row = frame.rows;
	int maxLen = MAX(col, row);
	Mat netInputImg = frame.clone();
	if (maxLen > 1.2*col || maxLen > 1.2*row) {
		Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
		frame.copyTo(resizeImg(Rect(0, 0, col, row)));
		netInputImg = resizeImg;
	}
	this->src_height = netInputImg.rows;
	this->src_width = netInputImg.cols;
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(this->inpWidth, this->inpHeight), cv::Scalar(104, 117,123), true, false);
	this->net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	this->net.forward(netOutputImg, net.getUnconnectedOutLayersNames());

	for (int c = 0; c < netOutputImg.size(); c++)
	{
		std::cout << "i: " << c << " shape: " << netOutputImg[c].size[1] << " " << netOutputImg[c].size[2] << " " \
	              << netOutputImg[c].size[3] << std::endl;
	}
	
	postprocess(netOutputImg, det_results);
}

// int YOLO::postprocess(std::vector<cv::Mat> &outputs, std::vector<DetResults> &det_results)
// {
// 	int rval = 0;

// 	std::vector<int> classIds;//结果id数组
// 	std::vector<float> confidences;//结果每个id对应置信度数组
// 	std::vector<cv::Rect> boxes;//每个id矩形框
// 	float ratio_h = (float)this->src_height / this->inpHeight;
// 	float ratio_w = (float)this->src_width / this->inpWidth;
// 	int net_width = this->classes.size() + 5;  //输出的网络宽度是类别数+5
// 	// float* pdata = (float*)outputs[0].data;
// 	for (int stride =0; stride < 3; stride++) 
// 	{    //stride
// 		float* pdata = (float*)outputs[stride].data;
// 		int grid_x = (int)(this->inpWidth / this->stride[stride]);
// 		int grid_y = (int)(this->inpHeight / this->stride[stride]);
// 		for (int anchor = 0; anchor < 3; anchor++) { //anchors
// 			const float anchor_w = this->anchors[stride][anchor * 2];
// 			const float anchor_h = this->anchors[stride][anchor * 2 + 1];
// 			for (int i = 0; i < grid_y; i++) {
// 				for (int j = 0; j < grid_x; j++) {
// 					float box_score = pdata[4]; //Sigmoid(pdata[4]);//获取每一行的box框中含有某个物体的概率
// 					if (box_score > this->objThreshold) {
// 						std::cout << "box_score: " << stride << " " << anchor << " " << i << " " << j << " " << box_score << std::endl;
// 						cv::Mat scores(1, this->classes.size(), CV_32FC1, pdata + 5);
// 						Point classIdPoint;
// 						double max_class_socre;
// 						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
// 						max_class_socre = (float)max_class_socre; //Sigmoid((float)max_class_socre);
// 						if (max_class_socre > this->confThreshold) {
// 							//rect [x,y,w,h]
// 							float x = pdata[0];  //x // (sigmoid_x(pdata[0]) * 2.f - 0.5f + j) * this->stride[stride];
// 							float y = pdata[1]; //y  // (sigmoid_x(pdata[1]) * 2.f - 0.5f + i) * this->stride[stride];   
// 							float w = pdata[2]; //w // powf(sigmoid_x(pdata[2]) * 2.f, 2.f) * anchor_w;  
// 							float h = pdata[3]; //h // powf(sigmoid_x(pdata[3]) * 2.f, 2.f) * anchor_h; 
// 							int left = (x - 0.5*w)*ratio_w;
// 							int top = (y - 0.5*h)*ratio_h;
// 							classIds.push_back(classIdPoint.x);
// 							confidences.push_back(max_class_socre*box_score);
// 							boxes.push_back(Rect(left, top, int(w*ratio_w), int(h*ratio_h)));
// 						}
// 					}
// 					pdata += net_width;//下一行
// 				}
// 			}
// 		}
// 	}

// 	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
// 	vector<int> nms_result;
// 	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, nms_result);
// 	for (int i = 0; i < nms_result.size(); i++) {
// 		int idx = nms_result[i];
// 		DetResults result;
// 		result.class_id = classIds[idx];
// 		result.confidence = confidences[idx];
// 		result.box = boxes[idx];
// 		det_results.push_back(result);
// 	}

// 	return rval;
// }

int YOLO::postprocess(std::vector<cv::Mat> &outputs, std::vector<DetResults> &det_results)
{
	int rval = 0;

	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	float ratio_h = (float)this->src_height / this->inpHeight;
	float ratio_w = (float)this->src_width / this->inpWidth;
	int net_width = this->classes.size() + 5;  //输出的网络宽度是类别数+5

	for (int stride =0; stride < 3; stride++) 
	{    //stride
		float* pdata = (float*)outputs[stride].data;
		int grid_x = (int)(this->inpWidth / this->stride[stride]);
		int grid_y = (int)(this->inpHeight / this->stride[stride]);
		float max_class_socre;
    	int cls = 0;
		for (int anchor = 0; anchor < 3; anchor++) { //anchors
			const float anchor_w = this->anchors[stride][anchor * 2];
			const float anchor_h = this->anchors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_y; i++) {
				for (int j = 0; j < grid_x; j++) {
					float box_score = sigmoid_x(pdata[IDX(4)]);//获取每一行的box框中含有某个物体的概率
					if (box_score > this->objThreshold) {
						// cv::Mat scores(1, this->classes.size(), CV_32FC1, pdata + 5);
						// Point classIdPoint;
						// double max_class_socre;
						// minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre = -1e10;
						cls = 0;
						for (int p = 0; p < g_classificationCnt; p++) {
							if (pdata[IDX(5 + p)] >= max_class_socre) {
								max_class_socre = pdata[IDX(5 + p)];
								cls = p;
							}
						}
						
						max_class_socre = sigmoid_x(max_class_socre);
						box_score *= max_class_socre;

						// max_class_socre = sigmoid_x((float)max_class_socre); // (float)max_class_socre;
						if (box_score > this->confThreshold) {
							// rect [x,y,w,h]
							float x = (sigmoid_x(pdata[IDX(0)]) * 2.f - 0.5f + j) * this->stride[stride]; //x pdata[0];   // 
							float y = (sigmoid_x(pdata[IDX(1)]) * 2.f - 0.5f + i) * this->stride[stride]; //y pdata[1]; //    
							float w = powf(sigmoid_x(pdata[IDX(2)]) * 2.f, 2.f) * anchor_w; //w pdata[2]; //  
							float h = powf(sigmoid_x(pdata[IDX(3)]) * 2.f, 2.f) * anchor_h; //h pdata[3]; //  
							int left = (x - 0.5*w)*ratio_w;
							int top = (y - 0.5*h)*ratio_h;
							classIds.push_back(cls);
							confidences.push_back(box_score);
							boxes.push_back(Rect(left, top, int(w*ratio_w), int(h*ratio_h)));
						}
					}
					// pdata += net_width;//下一行
				}
			}
		}
	}

	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, nms_result);
	for (int i = 0; i < nms_result.size(); i++) {
		int idx = nms_result[i];
		DetResults result;
		result.class_id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		det_results.push_back(result);
	}

	return rval;
}

int main(int argc, char** argv)
{
	YOLO yolo_model(yolo_nets[0]);

    if( argc != 2)
	{
        printf("usage: ./yolov5 image_dir\n");
        exit(0);
    }
	std::vector<std::string> images;
	ListImages(argv[1], images);
    std::cout << "total Test images : " << images.size() << std::endl;

	for (size_t index = 0; index < images.size(); index++) 
	{
		std::stringstream temp_str;
     	temp_str << argv[1] << images[index];
		std::cout << temp_str.str() << std::endl;
		Mat srcimg = imread(temp_str.str());
		std::vector<DetResults> det_results;
		yolo_model.run(srcimg, det_results);

#ifdef IS_SHOW
		yolo_model.drawPred(srcimg, det_results);
		static const string kWinName = "Deep learning object detection in OpenCV";
		namedWindow(kWinName); // WINDOW_NORMAL
		imshow(kWinName, srcimg);
		waitKey(0);
		destroyAllWindows();
#endif

#ifdef IS_SAVE_TXT
		stringstream out_file_name;
		out_file_name << "./labels/" << images[index].replace(images[index].find("."), 4, ".txt");
		std::cout << out_file_name.str() << std::endl;
		ofstream labelfile(out_file_name.str(), ios::out);	
		yolo_model.save_txt(labelfile, det_results);
		labelfile.close();
#endif
	}
	return 0;
}

// int main() {
// 	std::string model_file="../yolov5s.onnx";
// 	cv::dnn::Net net = cv::dnn::readNetFromONNX(model_file);

// 	Mat srcimg = imread("../273271,22f4f0002b8b12fd.jpg");
// 	Mat blob;
// 	blobFromImage(srcimg, blob, 1 / 255.0, Size(640, 640), Scalar(0, 0, 0), true, false);
// 	net.setInput(blob, "images");
// 	vector<Mat> outs;
// 	Mat out_1;
// 	Mat out_2;
// 	Mat out_3;
// 	net.forward(outs, net.getUnconnectedOutLayersNames());
// 	// outs = net.forward("326", "385", "444");

// 	std::cout << outs.size() << std::endl;

// 	// for (int i = 0; i < outs.size(); i++)
// 	// {
// 	// 	std::cout << "i: " << i << " shape: " << outs[i].size[1] << " " << outs[i].size[2] << " " << outs[i].size[3] << std::endl;
// 	// }
// }