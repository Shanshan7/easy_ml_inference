#include <vector>
#include <stdlib.h>
#include <iostream>

#include "onnxruntime_cxx_api.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"


using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
    // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0));
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    const char* model_path = "mobilenetv2-1.0.onnx";
    Ort::Session session(env, model_path, session_options);
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    size_t num_input_nodes = session.GetInputCount();
    std::vector<const char*> input_node_names = {"data"};
    std::vector<const char*> output_node_names = {"mobilenetv20_output_flatten0_reshape0"};

    static constexpr const int width_ = 224;
    static constexpr const int height_ = 224;
    static constexpr const int channel = 3;

    std::array<float, width_ * height_*channel> input_image_{};
    Ort::Value input_tensor_{ nullptr };
    std::array<int64_t, 4> input_shape_{ 1,3, width_, height_ };
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), 
                                                    input_image_.size(), input_shape_.data(), input_shape_.size());

    Mat img = imread("./cls_001.jpg");
    const int row = 224;
    const int col = 224;
    Mat dst(row, col, CV_8UC3);
    Mat dst2;
    resize(img, dst, Size(row, col));
    cvtColor(dst, dst, COLOR_BGR2RGB);

    float* output = input_image_.data();
    fill(input_image_.begin(), input_image_.end(), 0.f);
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (c == 0) {
                    output[c*row*col + i * col + j] = ((dst.ptr<uchar>(i)[j * 3 + c]) / 255.0 - 0.406) / 0.225;
                }
                if (c == 1) {
                    output[c*row*col + i * col + j] = ((dst.ptr<uchar>(i)[j * 3 + c]) / 255.0 - 0.456) / 0.224;
                }
                if (c == 2) {
                    output[c*row*col + i * col + j] = ((dst.ptr<uchar>(i)[j * 3 + c]) / 255.0 - 0.485) / 0.229;
                }
            }
        }
    }

    assert(input_tensor_.IsTensor());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor_));
    // score model & input tensor, get back output tensor
    double timeStart = (double)getTickCount();
    for (int i = 0; i < 1000; i++) {
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), 
                                        ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 1);
    }
    double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
    cout << "running time : " << nTime << "sec\n" << endl;

    // Get pointer to output tensor float values
    // float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    // std::cout << floatarr[0] << std::endl;

    printf("Done!\n");
    return 0;
}

