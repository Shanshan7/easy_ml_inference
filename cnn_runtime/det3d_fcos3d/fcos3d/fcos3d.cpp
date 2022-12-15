#include <fstream>
#include <memory>
#include <math.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "fcos3d.h"

#define IMAGE_H 1080
#define IMAGE_W 1920
#define INPUT_H 1080
#define INPUT_W 1920
#define OUTPUT_H (INPUT_H / 4)
#define OUTPUT_W (INPUT_W / 4)
#define NUM_CLASSES 10
#define SCORE_THRESH 0.2f
#define NMS_THRESH 0.45f
#define DIR_OFFSET 0.7854f
#define TOPK 3645
#define PI 3.1415926535897932


FCOS3D::FCOS3D(const std::string& engine_path)
{
    buffer_size_[0] = 3 * INPUT_H * INPUT_W;
    buffer_size_[1] = TOPK * 3;
    buffer_size_[2] = TOPK * 9;
    buffer_size_[3] = TOPK * 1;
    buffer_size_[4] = TOPK * 10;
    buffer_size_[5] = TOPK * 1;

    cudaMalloc(&buffers_[0], buffer_size_[0] * sizeof(float));
    cudaMalloc(&buffers_[1], buffer_size_[1] * sizeof(float));
    cudaMalloc(&buffers_[2], buffer_size_[2] * sizeof(float));
    cudaMalloc(&buffers_[3], buffer_size_[3] * sizeof(int));
    cudaMalloc(&buffers_[4], buffer_size_[4] * sizeof(float));
    cudaMalloc(&buffers_[5], buffer_size_[5] * sizeof(float));

    image_data_.resize(buffer_size_[0]);
    mlvl_centers2d_temp.resize(buffer_size_[1]);
    mlvl_bboxes_temp.resize(buffer_size_[2]);
    mlvl_dir_scores.resize(buffer_size_[3]);
    mlvl_scores_temp.resize(buffer_size_[4]);
    mlvl_centerness.resize(buffer_size_[5]);
    mlvl_centers2d.resize(TOPK);
    mlvl_bboxes.resize(TOPK);
    mlvl_scores.resize(TOPK);
    std::cout << "mlvl_scores_size: " << mlvl_scores.size() << std::endl;

    cudaStreamCreate(&stream_);
    LoadEngine(engine_path);

    // Modify camera intrinsics due to scaling and crop
    // std::cout << intrinsic_.at<float>(1, 2) << std::endl;
    // intrinsic_.at<float>(1, 2) = intrinsic_.at<float>(1, 2) - (1080-680);
    cam_to_img(0, 0) = cam_to_img(0, 0) * INPUT_W / IMAGE_W;
    cam_to_img(0, 2) = cam_to_img(0, 2) * INPUT_W / IMAGE_W;
    cam_to_img(1, 1) = cam_to_img(1, 1) * INPUT_W / IMAGE_W;
    cam_to_img(1, 2) = cam_to_img(1, 2) * INPUT_W / IMAGE_W;
}

FCOS3D::~FCOS3D() 
{
    cudaStreamDestroy(stream_);
    for (auto& buffer : buffers_) {
        cudaFree(buffer);
    }
    if (context_ != nullptr) {
        context_->destroy();
        engine_->destroy();
    }
}

void FCOS3D::detect(const cv::Mat& raw_img) 
{
    // Preprocessing
    // cv::Rect roi_rect(0, 400, 1920, 680);
    // cv::Mat img_crop = raw_img(roi_rect).clone();

    float divisor = 32.0;
    const int pad_h = int(ceil(raw_img.rows / divisor)) * divisor;
    const int pad_w = int(ceil(raw_img.cols / divisor)) * divisor;
    // std::cout << "pad_w, pad_h: " << pad_w << " " << pad_h << std::endl;

    const int top = 0;
    const int bottom = pad_h - raw_img.rows;
    const int left = 0;
    const int right = pad_w - raw_img.cols;
    // std::cout << "pad: " << top << " " << bottom << " " << left << " " << right << std::endl;
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
    // std::cout << "image: " << data_chw << std::endl;

    // Do inference
    cudaMemcpyAsync(buffers_[0], image_data_.data(), buffer_size_[0] * sizeof(float), cudaMemcpyHostToDevice, stream_);
    context_->executeV2(&buffers_[0]);
    cudaMemcpyAsync(mlvl_centers2d_temp.data(), buffers_[1], buffer_size_[1] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(mlvl_bboxes_temp.data(), buffers_[2], buffer_size_[2] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(mlvl_dir_scores.data(), buffers_[3], buffer_size_[3] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(mlvl_scores_temp.data(), buffers_[4], buffer_size_[4] * sizeof(int), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(mlvl_centerness.data(), buffers_[5], buffer_size_[5] * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    // cudaStreamSynchronize(stream_);

    // transform temp data to struct data
    std::cout << "size: " << mlvl_centers2d.size() << std::endl;
    for(int l = 0; l < mlvl_centers2d.size(); l++)
    {
        Eigen::Vector3f centers2d;
        centers2d << mlvl_centers2d_temp[l*3], mlvl_centers2d_temp[l*3+1], \
                     mlvl_centers2d_temp[l*3+2];
        mlvl_centers2d[l] = centers2d;

        BboxDim bbox_dim;
        bbox_dim.x = mlvl_bboxes_temp[l*9];
        bbox_dim.y = mlvl_bboxes_temp[l*9+1];
        bbox_dim.depth = mlvl_bboxes_temp[l*9+2];
        bbox_dim.w = mlvl_bboxes_temp[l*9+3];
        bbox_dim.h = mlvl_bboxes_temp[l*9+4];
        bbox_dim.l = mlvl_bboxes_temp[l*9+5];
        bbox_dim.rotation = mlvl_bboxes_temp[l*9+6];
        bbox_dim.velocity_x = mlvl_bboxes_temp[l*9+7];
        bbox_dim.velocity_y = mlvl_bboxes_temp[l*9+8];
        mlvl_bboxes[l] = bbox_dim;

        std::vector<float> score;
        score.resize(10);
        for(int n = 0; n < NUM_CLASSES; n++)
        {
            score[n] = mlvl_scores_temp[l*10+n];
        }
        mlvl_scores[l] = score;
    }
    std::cout << "size: " << mlvl_scores_temp.size() << " " << mlvl_scores.size() << " " << mlvl_scores[0].size() << std::endl;

    // Decoding and visualization
    PostProcess();
}

void FCOS3D::LoadEngine(const std::string& engine_path) 
{
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

void FCOS3D::PostProcess() 
{
    // cls_scores: 5 * FPN feature map, 5种类型的输出（类别-10类，预测框-9个，方向类别-2个，属性-9个，中心点-1个）
    // 预测框：几何中心坐标、长宽高以及朝向角，速度vx,vy(offset, depth, size, rot, velo)
    // 正样本点越靠近真实边界框的中心，那么center-ness的值越接近1，越远离真实边界框的中心，越接近0
    // 类别置信度，方向类别预测，方向类别置信度，属性预测，属性置信度，中心点置信度，预测框
    // 1. 输出每个feature中score最高的nms_pre的所有目标物体信息
    // 2. change the offset to actual center predictions
    // 3. 获取mlvl_centers2d, mlvl_bboxes, mlvl_dir_scores进行下一步
    // 4. change local yaw to global yaw for 3D nms
    // 5. bev nms进行极大值抑制
    // 6. 最终输出框（相机坐标系下的9个点），得分，类别，属性

    float score_thr = 0.2;
    float nms_thr = 0.45;
    float dir_offset = 0.7854;

    PointsImg2Cam();
    std::cout << "1. --PointsImg2Cam--" << std::endl;
    // mlvl_bboxes[:, :3] = points_img2cam(mlvl_bboxes[:, :3], view)
    // change local yaw to global yaw for 3D nms
    DecodeYaw(dir_offset);
    std::cout << "2. --DecodeYaw--" << std::endl;

    // 2D BEV box of each box with rotation in XYWHR format, in shape (N, 5).
    std::vector<RotatedBox<float>> mlvl_bboxes_for_nms;
    mlvl_bboxes_for_nms.resize(mlvl_bboxes.size());
    for(int i = 0; i < mlvl_bboxes.size(); i++)
    {
        mlvl_bboxes_for_nms[i].x_ctr = mlvl_bboxes[i].x;
        mlvl_bboxes_for_nms[i].y_ctr = mlvl_bboxes[i].depth;
        mlvl_bboxes_for_nms[i].w = mlvl_bboxes[i].w;
        mlvl_bboxes_for_nms[i].h = mlvl_bboxes[i].h;
        mlvl_bboxes_for_nms[i].a = -mlvl_bboxes[i].rotation;
    }
    std::cout << "3. --BEV--" << std::endl;

    std::vector<std::vector<float>> mlvl_nms_scores;
    for(int j = 0; j < mlvl_scores.size(); j++)
    {
        std::vector<float> mlvl_nms_scores_vec;
        for(int k = 0; k < mlvl_scores[j].size(); k++)
        {
            float num_score = mlvl_scores[j][k] * mlvl_centerness[j];
            mlvl_nms_scores_vec.push_back(num_score);
        }
        mlvl_nms_scores.push_back(mlvl_nms_scores_vec);
    }
    std::cout << "4. --mlvl_nms_scores_vec--" << std::endl;

    Box3dMultiClassNms(mlvl_bboxes_for_nms,
                       mlvl_nms_scores,
                       score_thr,
                       nms_thr);
    std::cout << "5. --Box3dMultiClassNms--" << std::endl;
    for(int l = 0; l < result_bboxes.size(); l++)
    {
        result_bboxes[l].y = result_bboxes[l].y + result_bboxes[l].h * 0.5;
    }
    std::cout << "6. --resultbboxes--" << std::endl;
}

void FCOS3D::PointsImg2Cam() 
{
    for(int i = 0; i < mlvl_bboxes.size(); i++)
    {
        Eigen::Vector4f point;
        Eigen::Matrix4f homo_cam2img = Eigen::Matrix4f::Identity();

        // xys plus depth
        point << mlvl_bboxes[i].x * mlvl_bboxes[i].depth, mlvl_bboxes[i].y * mlvl_bboxes[i].depth, \
                 mlvl_bboxes[i].depth, 1.0;
        homo_cam2img.block<3,3>(0,0) = cam_to_img;

        // Do operation in homogeneous coordinates.
        Eigen::Vector4f points3D = homo_cam2img * point;
        mlvl_bboxes[i].x = points3D(0);
        mlvl_bboxes[i].y = points3D(1);
        mlvl_bboxes[i].depth = points3D(2);
    }
}

void FCOS3D::DecodeYaw(float dir_offset) 
{
    Eigen::Matrix4f homo_cam2img = Eigen::Matrix4f::Identity();
    homo_cam2img.block<3,3>(0,0) = cam_to_img;
    if(mlvl_bboxes.size() > 0)
    {
        for(int i = 0; i < mlvl_bboxes.size(); i++)
        {
            float val = mlvl_bboxes[i].rotation - dir_offset;
            float dir_rot = val - floor(val / PI + 0) * PI;
            float bbox_rotation = dir_rot + dir_offset + PI * mlvl_dir_scores[i];
            mlvl_bboxes[i].rotation = atan2(mlvl_centers2d[i](0) - homo_cam2img(0, 2),
                                homo_cam2img(0, 0)) + bbox_rotation;
        }
    }
}

void FCOS3D::Box3dMultiClassNms(std::vector<RotatedBox<float>> mlvl_bboxes_for_nms,
                                std::vector<std::vector<float>> mlvl_scores,
                                float score_thr,
                                float nms_thr) 
{
    // do multi class nms
    for(int i = 0; i < NUM_CLASSES; i++)
    {
        std::vector<int> indices;
        indices.clear();
        std::vector<float> score_index_vec;
        score_index_vec.clear();
        std::vector<int> dir_scores_vec;
        dir_scores_vec.clear();
        std::vector<BboxDim> bboxes_index_vec;
        bboxes_index_vec.clear();
        std::vector<RotatedBox<float>> bboxes_bev_index_vec;
        bboxes_bev_index_vec.clear();
        std::cout << "(0) ------------------------" << std::endl;

        // get bboxes and scores of this class
        std::cout << "size: " << mlvl_bboxes_for_nms.size() << " " << mlvl_scores[0].size() << std::endl;
        for(int j = 0; j < mlvl_bboxes.size(); j++)
        {
            if (mlvl_scores[j][i] > score_thr) {
                score_index_vec.push_back(mlvl_scores[j][i]);
                bboxes_index_vec.push_back(mlvl_bboxes[j]);
                std::cout << "bbox: " << mlvl_bboxes[j].x << std::endl;
                dir_scores_vec.push_back(mlvl_dir_scores[j]);
                std::cout << "dir_scores_vec: " << mlvl_dir_scores[j] << std::endl;
                bboxes_bev_index_vec.push_back(mlvl_bboxes_for_nms[j]);
                std::cout << "bboxes_bev_index_vec: " << bboxes_bev_index_vec[j].x_ctr << std::endl;
            }
            apply_nms_fast(bboxes_bev_index_vec, score_index_vec, score_thr, nms_thr, 0.01, 
                       bboxes_index_vec.size(), &indices);
#ifdef DEBUG
            std::cout << "indices value" << std::endl;
            for(int i = 0; i < indices.size(); i++)
            {
                std::cout << i << " " << indices[i] << std::endl;
            }
#endif
        }
        std::cout << "(2) apply_nms_fast" << std::endl;
        for(int k = 0; k < indices.size(); k++)
        {
            result_bboxes.push_back(bboxes_index_vec[k]);
            result_scores.push_back(score_index_vec[k]);
            result_labels.push_back(i);
            result_dir_scores.push_back(dir_scores_vec[k]);
        }
        std::cout << "(2) result_push_out" << std::endl;
    }
}