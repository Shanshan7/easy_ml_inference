#include "yolov5.h"

YOLO::YOLO(string model_path)
{
	// cout << "Net use " << config.netname << endl;
	// this->confThreshold = config.confThreshold;
	// this->nmsThreshold = config.nmsThreshold;
	// this->objThreshold = config.objThreshold;
	// strcpy(this->netname, config.netname.c_str());

	ifstream ifs(this->classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->classes.push_back(line);

	// string modelFile = this->netname;
	// modelFile += ".onnx";
	this->net = cv::dnn::readNet(model_path);
}

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

void YOLO::run(cv::Mat& frame, std::vector<DetectBox> &det_results)
{
	cv::Mat blob;
	int col = frame.cols;
	int row = frame.rows;
	int maxLen = max(col, row);
	cv::Mat netInputImg = frame.clone();
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
	cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(this->inpWidth, this->inpHeight), cv::Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	std::vector<cv::Mat> netOutputImg;
	this->net.forward(netOutputImg, net.getUnconnectedOutLayersNames());
	
	postprocess(netOutputImg, det_results);
}

int YOLO::postprocess(std::vector<cv::Mat> &outputs, std::vector<DetectBox> &det_results)
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
		DetectBox result;
		result.classID = boxes[i][4];
		// if (result.classID == 0) 
		// {
		result.confidence = boxes[i][5];
		result.x1 = boxes[i][0];
		// if (result.x1 < 1) {continue;}
		result.y1 = boxes[i][1];
		// if (result.y1 < 1) {continue;}
		result.x2 = boxes[i][0] + boxes[i][2];
		// if (result.x2 >= this->src_width) {result.x2 = this->src_width - 1;}
		result.y2 = boxes[i][1] + boxes[i][3];
		// if (result.y2 >= 1080) {result.y2 = 1080 - 1;}
		// std::cout << result.x1 << " " << result.y1 << " " << result.x2 << " " << result.y2 << std::endl;
		det_results.push_back(result);
		// }
	}

	return rval;
}