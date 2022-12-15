#pragma once

#include <iostream>
#include <NvInfer.h>
#include <opencv2/core.hpp>
#include <math.h>

//Eigen
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "box_iou_rotated_utils.h"
#include "nms_rotated_cpu.h"
// #include "utils.h"


class Logger : public nvinfer1::ILogger {
  public:
    explicit Logger(Severity severity = Severity::kWARNING)
        : reportable_severity(severity) {}

    void log(Severity severity, const char* msg) noexcept {
        if (severity > reportable_severity) {
            return;
        }
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "UNKNOWN: ";
                break;
        }
        std::cerr << msg << std::endl;
    }
    Severity reportable_severity;
};

struct BboxDim {
    float x;
    float y;
    float depth;
    float w;
    float h;
    float l;
    float rotation;
    float velocity_x;
    float velocity_y;
};

// struct BboxDimBEV {
//     float x;
//     float y;
//     float w;
//     float h;
//     float rotation;
// };

class FCOS3D {
  public:
    FCOS3D(const std::string& engine_path, const cv::Mat& intrinsic);
          
    ~FCOS3D();

    void detect(const cv::Mat& raw_img);
    // void getObjects(std::vector<DetectStruct> &detects, 
    //                    Eigen::Matrix4d &rt_lidar_to_cam);

  private:
    void LoadEngine(const std::string& engine_path);

    void PostProcess();
    void PointsImg2Cam();
    void DecodeYaw(float dir_offset);
    void Box3dMultiClassNms(std::vector<RotatedBox<float>> mlvl_bboxes_for_nms,
                        std::vector<std::vector<float>> mlvl_scores,
                        float score_thr,
                        float nms_thr);

    Logger g_logger_;
    cudaStream_t stream_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    
    void* buffers_[6];
    int buffer_size_[2];
    std::vector<float> image_data_;
    std::vector<float> mlvl_centers2d_temp;
    std::vector<float> mlvl_bboxes_temp;
    std::vector<float> mlvl_scores_temp;

    std::vector<Eigen::Vector3f> mlvl_centers2d;
    std::vector<BboxDim> mlvl_bboxes;
    std::vector<std::vector<float>> mlvl_scores;
    std::vector<int> mlvl_dir_scores;
    std::vector<float> mlvl_centerness;
    // cv::Mat intrinsic_;
    Eigen::Matrix3f cam_to_img;

    std::vector<BboxDim> result_bboxes;
    std::vector<float> result_scores;
    std::vector<int> result_dir_scores;
    std::vector<int> result_labels;
};