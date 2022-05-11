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

// void YOLO::get_square_size(const cv::Size src_size, const cv::Size dst_size, float &ratio, cv::Size &pad_size)
// {
//     const int src_width = src_size.width;
//     const int src_height = src_size.height;
//     const int dst_width = dst_size.width;
//     const int dst_height = dst_size.height;
//     ratio = std::min(static_cast<float>(dst_width) / src_width, \
//                      static_cast<float>(dst_height) / src_height);
//     const int new_width = static_cast<int>(src_width * ratio);
//     const int new_height = static_cast<int>(src_height * ratio);
//     const int pad_width = dst_width - new_width;
//     const int pad_height = dst_height - new_height;
//     pad_size.width = pad_width;
//     pad_size.height = pad_height;
// }

// void YOLO::image_resize_square(const cv::Mat &src, const cv::Size dst_size, cv::Mat &dst_image)
// {
//     const int src_width = src.cols;
//     const int src_height = src.rows;
//     float ratio;
//     cv::Size pad_size;
//     get_square_size(cv::Size(src_width, src_height), dst_size, ratio, pad_size);
//     int new_width = static_cast<int>(src_width * ratio);
//     int new_height = static_cast<int>(src_height * ratio);
//     const int top = pad_size.height / 2;
//     const int bottom = pad_size.height - (pad_size.height / 2);
//     const int left = pad_size.width / 2;
//     const int right = pad_size.width - (pad_size.width / 2);
//     cv::Mat resize_mat;
//     cv::resize(src, resize_mat, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
//     cv::copyMakeBorder(resize_mat, dst_image, top, bottom, left, right, cv::INTER_LINEAR, cv::Scalar(114, 114, 114));
// }

static float overlap(float x1, float w1, float x2, float w2)
{
    float left = std::max(x1 - w1 / 2.0f, x2 - w2 / 2.0f);
    float right = std::min(x1 + w1 / 2.0f, x2 + w2 / 2.0f);
    return right - left;
}

static float cal_iou(std::vector<float> box, std::vector<float>truth)
{
    float w = overlap(box[0], box[2], truth[0], truth[2]);
    float h = overlap(box[1], box[3], truth[1], truth[3]);
    if(w < 0 || h < 0) return 0;

    float inter_area = w * h;
    float union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area;
    return inter_area * 1.0f / union_area;
}

/**************************************
Parammeter: 
            1.boxes: all detection objects
            2.classes: classes number(unuse in this version)
            3.thres: iou threshold
            4.sign_nms: do road sign nms or not, default is true

Return: detection objects after nms
**************************************/
std::vector<std::vector<float>> YOLO::applyNMS(std::vector<std::vector<float>>& boxes,
	                                    const float thres) 
{    
    std::vector<std::vector<float>> result;
    std::vector<bool> exist_box(boxes.size(), true);

    int n = 0;
    for (size_t _i = 0; _i < boxes.size(); ++_i) 
	{
        if (!exist_box[_i]) 
			continue;
        n = 0;
        for (size_t _j = _i + 1; _j < boxes.size(); ++_j)
		{
            // different class name
            if (!exist_box[_j] || boxes[_i][4] != boxes[_j][4]) 
				continue;
            float ovr = cal_iou(boxes[_j], boxes[_i]);
            if (ovr >= thres) 
            {
                if (boxes[_j][5] <= boxes[_i][5])
                {
                    exist_box[_j] = false;
                }
                else
                {
                    n++;   // have object_score bigger than boxes[_i]
                    exist_box[_i] = false;
                    break;
                }
            }
        }
        //if (n) exist_box[_i] = false;
		if (n == 0) 
		{
			result.push_back(boxes[_i]);
		}			
    }

    return result;
}

void YOLO::run(Mat& frame, std::vector<DetResults> &det_results)
{
	Mat blob;
	int col = frame.cols;
	int row = frame.rows;
	int maxLen = max(col, row);
	Mat netInputImg = frame.clone();
	// if (maxLen > 1.2*col || maxLen > 1.2*row) {
	// 	Mat resizeImg = Mat::ones(maxLen, maxLen, CV_8UC3) * 114;
	// 	frame.copyTo(resizeImg(Rect(0, 0, col, row)));
	// 	netInputImg = resizeImg;
	// }
    // float ratio = static_cast<float>(this->inpWidth) /  static_cast<float>(this->inpHeight);
	// const int new_height = std::max(static_cast<int>(col / ratio), row);
	// const int new_width = std::max(static_cast<int>(row * ratio), col);
	// Mat resizeImg = Mat::ones(new_height, new_width, CV_8UC3) * 114;
	// frame.copyTo(resizeImg(Rect(0, 0, col, row)));
	// netInputImg = resizeImg;
	this->src_height = netInputImg.rows;
	this->src_width = netInputImg.cols;
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(this->inpWidth, this->inpHeight), cv::Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	this->net.forward(netOutputImg, net.getUnconnectedOutLayersNames());

	// for (int c = 0; c < netOutputImg.size(); c++)
	// {
	// 	std::cout << "i: " << c << " shape: " << netOutputImg[c].size[1] << " " << netOutputImg[c].size[2] << " " \
	//               << netOutputImg[c].size[3] << std::endl;
	// }
	
	postprocess(netOutputImg, det_results);
}

int YOLO::postprocess(std::vector<cv::Mat> &outputs, std::vector<DetResults> &det_results)
{
	int rval = 0;

	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	// std::vector<cv::Rect> boxes;//每个id矩形框
	std::vector<std::vector<float>> boxes;
	float ratio_h = (float)this->src_height / this->inpHeight;
	float ratio_w = (float)this->src_width / this->inpWidth;
	int net_width = this->classes.size() + 5;  //输出的网络宽度是类别数+5

	for (int stride =0; stride < 3; stride++) 
	{    //stride
		std::vector<float> box;
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
							box.clear();
							float x = (sigmoid_x(pdata[IDX(0)]) * 2.f - 0.5f + j) * this->stride[stride]; //x pdata[0];   // 
							float y = (sigmoid_x(pdata[IDX(1)]) * 2.f - 0.5f + i) * this->stride[stride]; //y pdata[1]; //    
							float w = powf(sigmoid_x(pdata[IDX(2)]) * 2.f, 2.f) * anchor_w; //w pdata[2]; //  
							float h = powf(sigmoid_x(pdata[IDX(3)]) * 2.f, 2.f) * anchor_h; //h pdata[3]; //  
							int left = (x - 0.5*w)*ratio_w;
							int top = (y - 0.5*h)*ratio_h;
							classIds.push_back(cls);
							confidences.push_back(box_score);
							box.push_back(float(left));
							box.push_back(float(top));
							box.push_back(w*ratio_w);
							box.push_back(h*ratio_h);
							box.push_back(float(cls));
							box.push_back(float(box_score));
							boxes.push_back(box);
						}
					}
				}
			}
		}
	}

	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	// vector<int> nms_result;
	// NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, nms_result);
	boxes = applyNMS(boxes, this->nmsThreshold);
	for (int i = 0; i < boxes.size(); i++) {
		DetResults result;
		result.class_id = boxes[i][4];
		result.confidence = boxes[i][5];
		result.box = Rect(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]);
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
		namedWindow("image", 0);
		resizeWindow("image", srcimg.cols*0.4, srcimg.rows*0.4);
		imshow("image", srcimg);
		waitKey(0);
		// destroyAllWindows();
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