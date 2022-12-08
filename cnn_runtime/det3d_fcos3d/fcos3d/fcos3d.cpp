#include <fstream>
#include <memory>
#include <math.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "fcos3d.h"

// #define IMAGE_H 375
// #define IMAGE_W 1242
#define IMAGE_H 1080
#define IMAGE_W 1920
// #define IMAGE_H 900
// #define IMAGE_W 1600
#define INPUT_H 1080
#define INPUT_W 1920
#define OUTPUT_H (INPUT_H / 4)
#define OUTPUT_W (INPUT_W / 4)
#define SCORE_THRESH 0.3f
#define TOPK 100


FCOS3D::FCOS3D(const std::string& engine_path, const cv::Mat& intrinsic)
{
    buffer_size_[0] = 3 * INPUT_H * INPUT_W;
    buffer_size_[1] = TOPK * 8;
    buffer_size_[2] = TOPK;
    buffer_size_[3] = TOPK;
    cudaMalloc(&buffers_[0], buffer_size_[0] * sizeof(float));
    cudaMalloc(&buffers_[1], buffer_size_[1] * sizeof(float));
    cudaMalloc(&buffers_[2], buffer_size_[2] * sizeof(float));
    cudaMalloc(&buffers_[3], buffer_size_[3] * sizeof(float));
    image_data_.resize(buffer_size_[0]);
    bbox_preds_.resize(buffer_size_[1]);
    topk_scores_.resize(buffer_size_[2]);
    topk_indices_.resize(buffer_size_[3]);
    cudaStreamCreate(&stream_);
    LoadEngine(engine_path);

    // // https://github.com/open-mmlab/mmdetection3d/blob/master/configs/_base_/models/smoke.py#L41
    // base_depth_ = {28.01f, 16.32f};
    // base_dims_.resize(3);  //pedestrian, cyclist, car
    // base_dims_[0].x = 0.88f;
    // base_dims_[0].y = 1.73f;
    // base_dims_[0].z = 0.67f;
    // base_dims_[1].x = 1.78f;
    // base_dims_[1].y = 1.70f;
    // base_dims_[1].z = 0.58f;
    // base_dims_[2].x = 5.88f; // 3.88
    // base_dims_[2].y = 2.13f; // 2.63
    // base_dims_[2].z = 2.03f; // 2.53
    // // Modify camera intrinsics due to scaling
    // std::cout << intrinsic_.at<float>(1, 2) << std::endl;
    // intrinsic_.at<float>(1, 2) = intrinsic_.at<float>(1, 2) - (1080-680);
    
    // intrinsic_.at<float>(0, 0) *= static_cast<float>(INPUT_W) / IMAGE_W;
    // intrinsic_.at<float>(0, 2) *= static_cast<float>(INPUT_W) / IMAGE_W;
    // intrinsic_.at<float>(1, 1) *= static_cast<float>(INPUT_H) / IMAGE_H;
    // intrinsic_.at<float>(1, 2) *= static_cast<float>(INPUT_H) / IMAGE_H;
}

FCOS3D::~FCOS3D() {
    cudaStreamDestroy(stream_);
    for (auto& buffer : buffers_) {
        cudaFree(buffer);
    }
    if (context_ != nullptr) {
        context_->destroy();
        engine_->destroy();
    }
}

void FCOS3D::detect(const cv::Mat& raw_img) {
    // Preprocessing
    // cv::Rect roi_rect(0, 400, 1920, 680);
    // cv::Mat img_crop = raw_img(roi_rect).clone();
    // // cv::imshow("roi", img_crop);
    // // cv::waitKey();
    float divisor = 32.0;
    const int pad_h = int(ceil(raw_img.rows / divisor)) * divisor;
    const int pad_w = int(ceil(raw_img.cols / divisor)) * divisor;
    std::cout << "pad_w, pad_h: " << pad_w << " " << pad_h << std::endl;

    const int top = 0;
    const int bottom = pad_h - raw_img.rows;
    const int left = 0;
    const int right = pad_w - raw_img.cols;
    std::cout << "pad: " << top << " " << bottom << " " << left << " " << right << std::endl;
    cv::Mat dst_image;
    cv::copyMakeBorder(raw_img, dst_image, top, bottom, left, right, cv::INTER_LINEAR, cv::Scalar(103.530f, 116.280f, 123.675f));

    // cv::Mat img_resize;
    // cv::resize(img_crop, img_resize, cv::Size(INPUT_W, INPUT_H), cv::INTER_LINEAR);
    // img_resize.convertTo(img_resize, CV_32FC3, 1.0f);
    float mean[3] {103.530f, 116.280f, 123.675f};
    float std[3] = {1.0f, 1.0f, 1.0f};
    uint8_t* data_hwc = reinterpret_cast<uint8_t*>(dst_image.data);
    float* data_chw = image_data_.data();
    for (int c = 0; c < 3; ++c) {
        for (unsigned j = 0, img_size = INPUT_H * INPUT_W; j < img_size; ++j) {
            data_chw[c * img_size + j] = (data_hwc[j * 3 + c] - mean[c]) / std[c];  //no bgr2rgb
        }
    }
    std::cout << "image: " << data_chw << std::endl;

    // // Do inference
    // cudaMemcpyAsync(buffers_[0], image_data_.data(), buffer_size_[0] * sizeof(float), cudaMemcpyHostToDevice, stream_);
    // context_->executeV2(&buffers_[0]);
    // cudaMemcpyAsync(bbox_preds_.data(), buffers_[1], buffer_size_[1] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    // cudaMemcpyAsync(topk_scores_.data(), buffers_[2], buffer_size_[2] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    // cudaMemcpyAsync(topk_indices_.data(), buffers_[3], buffer_size_[3] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    // // cudaStreamSynchronize(stream_);

    // // Decoding and visualization
    // PostProcess(img_resize);
}

void FCOS3D::LoadEngine(const std::string& engine_path) {
    std::ifstream in_file(engine_path, std::ios::binary);
    if (!in_file.is_open()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return;
    }
    in_file.seekg(0, in_file.end);
    int length = in_file.tellg();
    in_file.seekg(0, in_file.beg);
    std::unique_ptr<char[]> trt_model_stream(new char[length]);
    in_file.read(trt_model_stream.get(), length);
    in_file.close();

    // getPluginCreator could not find plugin: MMCVModulatedDeformConv2d version: 1
    initLibNvInferPlugins(&g_logger_, "");
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger_);
    assert(runtime != nullptr);
    engine_ = runtime->deserializeCudaEngine(trt_model_stream.get(), length, nullptr);
    assert(engine_ != nullptr);
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);

    runtime->destroy();
}

void FCOS3D::PostProcess(cv::Mat& input_img) {
    // cls_scores: 5 * FPN feature map, 5种类型的输出（类别-10类，预测框-9个，方向类别-2个，属性-9个，中心点-1个）
    // 正样本点越靠近真实边界框的中心，那么center-ness的值越接近1，越远离真实边界框的中心，越接近0
    // 类别置信度，方向类别预测，方向类别置信度，属性预测，属性置信度，中心点，预测框
    // 1. 输出每个feature中score最高的nms_pre的所有目标物体信息
    // 2. change the offset to actual center predictions
    // 3. 获取mlvl_centers2d, mlvl_bboxes, mlvl_dir_scores进行下一步
    // 4. change local yaw to global yaw for 3D nms
    // 5. bev nms进行极大值抑制
    // 6. 最终输出框（相机坐标系下的9个点），得分，类别，属性

    for cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, \
                points in zip(cls_scores, bbox_preds, dir_cls_preds,
                              attr_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
            attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
            attr_score = torch.max(attr_pred, dim=-1)[1]
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1,
                                                     sum(self.group_reg_dims))
            bbox_pred = bbox_pred[:, :self.bbox_code_size]
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_pred = dir_cls_pred[topk_inds, :]
                centerness = centerness[topk_inds]
                dir_cls_score = dir_cls_score[topk_inds]
                attr_score = attr_score[topk_inds]
            # change the offset to actual center predictions
            bbox_pred[:, :2] = points - bbox_pred[:, :2]
            if rescale:
                bbox_pred[:, :2] /= bbox_pred[:, :2].new_tensor(scale_factor)
            pred_center2d = bbox_pred[:, :3].clone()
            bbox_pred[:, :3] = points_img2cam(bbox_pred[:, :3], view)
            mlvl_centers2d.append(pred_center2d)
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)
            mlvl_attr_scores.append(attr_score)
            mlvl_centerness.append(centerness)
}