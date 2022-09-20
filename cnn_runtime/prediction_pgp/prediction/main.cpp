// #include <vector>
// #include <stdlib.h>
// #include <iostream>

// #include "onnxruntime_cxx_api.h"
// #include "tensorrt_provider_factory.h"

// #include "opencv2/core.hpp"
// #include "opencv2/imgproc.hpp"
// #include "opencv2/imgcodecs.hpp"


// using namespace cv;
// using namespace std;

// int main(int argc, char* argv[]) {
//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
//     Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
//     // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0));
//     session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

//     const char* model_path = "./pgp.onnx";
//     Ort::Session session(env, model_path, session_options);
//     // print model input layer (node names, types, shape etc.)
//     Ort::AllocatorWithDefaultOptions allocator;

//     // print number of model input nodes
//     size_t num_input_nodes = session.GetInputCount();
//     std::vector<const char*> input_node_names = {"input.1", "input.7", "lane_node_masks", \
//                                                   "f4", "nbr_vehicle_masks", "f6", "nbr_ped_masks", \
//                                                   "f8", "f9", "9", "edge_type"};
//     std::vector<const char*> output_node_names = {"2180"};

//     // input.1
//     std::array<int64_t, 3> input_shape_{ 1,5,5 };
//     std::array<float, 5*5*1> input_image_{};
//     Ort::Value input_tensor_{ nullptr };
//     auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), 
//                                                     input_image_.size(), input_shape_.data(), input_shape_.size());
//     assert(input_tensor_.IsTensor());

//     // input.7
//     std::array<int64_t, 4> input_shape_7{ 1,164,20,6 };
//     std::array<float, 1*164*20*6> input_image_7{};
//     Ort::Value input_tensor_7{ nullptr };
//     auto allocator_info_7 = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     input_tensor_7 = Ort::Value::CreateTensor<float>(allocator_info_7, input_image_7.data(), 
//                                                     input_image_7.size(), input_shape_7.data(), input_shape_7.size());
//     assert(input_tensor_7.IsTensor());

//     // lane_node_masks
//     std::array<int64_t, 4> lane_node_masks_input_shape{ 1,164,20,6 };
//     std::array<float, 1*164*20*6> lane_node_masks_input_image{};
//     Ort::Value lane_node_masks_input_tensor{ nullptr };
//     auto lane_node_masks_allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     lane_node_masks_input_tensor = Ort::Value::CreateTensor<float>(lane_node_masks_allocator_info, lane_node_masks_input_image.data(), 
//                                                     lane_node_masks_input_image.size(), lane_node_masks_input_shape.data(), 
//                                                     lane_node_masks_input_shape.size());
//     assert(lane_node_masks_input_tensor.IsTensor());

//     // f4
//     std::array<int64_t, 4> f4_input_shape{ 1,84,5,5 };
//     std::array<float, 1*84*5*5> f4_input_image{};
//     Ort::Value f4_input_tensor{ nullptr };
//     auto f4_allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     f4_input_tensor = Ort::Value::CreateTensor<float>(f4_allocator_info, f4_input_image.data(), 
//                                                     f4_input_image.size(), f4_input_shape.data(), f4_input_shape.size());
//     assert(f4_input_tensor.IsTensor());

//     // nbr_vehicle_masks
//     std::array<int64_t, 4> nbr_vehicle_masks_input_shape{ 1,84,5,5 };
//     std::array<float, 1*84*5*5> nbr_vehicle_masks_input_image{};
//     Ort::Value nbr_vehicle_masks_input_tensor{ nullptr };
//     auto nbr_vehicle_masks_allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     nbr_vehicle_masks_input_tensor = Ort::Value::CreateTensor<float>(nbr_vehicle_masks_allocator_info, nbr_vehicle_masks_input_image.data(), 
//                                                     nbr_vehicle_masks_input_image.size(), nbr_vehicle_masks_input_shape.data(), 
//                                                     nbr_vehicle_masks_input_shape.size());
//     assert(nbr_vehicle_masks_input_tensor.IsTensor());

//     // f6
//     std::array<int64_t, 4> f6_input_shape{ 1,77,5,5 };
//     std::array<float, 1*77*5*5> f6_input_image{};
//     Ort::Value f6_input_tensor{ nullptr };
//     auto f6_allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     f6_input_tensor = Ort::Value::CreateTensor<float>(f6_allocator_info, f6_input_image.data(), 
//                                                     f6_input_image.size(), f6_input_shape.data(), f6_input_shape.size());
//     assert(f6_input_tensor.IsTensor());

//     // nbr_ped_masks
//     std::array<int64_t, 4> nbr_ped_masks_input_shape{ 1,77,5,5 };
//     std::array<float, 1*77*5*5> nbr_ped_masks_input_image{};
//     Ort::Value nbr_ped_masks_input_tensor{ nullptr };
//     auto nbr_ped_masks_allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     nbr_ped_masks_input_tensor = Ort::Value::CreateTensor<float>(nbr_ped_masks_allocator_info, nbr_ped_masks_input_image.data(), 
//                                                     nbr_ped_masks_input_image.size(), nbr_ped_masks_input_shape.data(), 
//                                                     nbr_ped_masks_input_shape.size());
//     assert(nbr_ped_masks_input_tensor.IsTensor());

//     // f8
//     std::array<int64_t, 3> f8_input_shape{ 1,164,84 };
//     std::array<float, 1*164*84> f8_input_image{};
//     Ort::Value f8_input_tensor{ nullptr };
//     auto f8_allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     f8_input_tensor = Ort::Value::CreateTensor<float>(f8_allocator_info, f8_input_image.data(), 
//                                                     f8_input_image.size(), f8_input_shape.data(), f8_input_shape.size());
//     assert(f8_input_tensor.IsTensor());

//     // f9
//     std::array<int64_t, 3> f9_input_shape{ 1,164,77 };
//     std::array<float, 1*164*77> f9_input_image{};
//     Ort::Value f9_input_tensor{ nullptr };
//     auto f9_allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     f9_input_tensor = Ort::Value::CreateTensor<float>(f9_allocator_info, f9_input_image.data(), 
//                                                     f9_input_image.size(), f9_input_shape.data(), f9_input_shape.size());
//     assert(f9_input_tensor.IsTensor());

//     // 9
//     std::array<int64_t, 3> input_shape_9{ 1,164,15 };
//     std::array<float, 1*164*15> input_image_9{};
//     Ort::Value input_tensor_9{ nullptr };
//     auto allocator_info_9 = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     input_tensor_9 = Ort::Value::CreateTensor<float>(allocator_info_9, input_image_9.data(), 
//                                                     input_image_7.size(), input_shape_9.data(), input_shape_9.size());
//     assert(input_tensor_9.IsTensor());

//     // edge_type
//     std::array<int64_t, 3> edge_type_input_shape{ 1,164,15 };
//     std::array<int64, 1*164*15> edge_type_input_image{};
//     Ort::Value edge_type_input_tensor{ nullptr };
//     auto edge_type_allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     edge_type_input_tensor = Ort::Value::CreateTensor<int64>(edge_type_allocator_info, edge_type_input_image.data(), 
//                                                     edge_type_input_image.size(), edge_type_input_shape.data(), 
//                                                     edge_type_input_shape.size());
//     assert(edge_type_input_tensor.IsTensor());

//     std::vector<Ort::Value> ort_inputs;
//     ort_inputs.push_back(std::move(input_tensor_));
//     ort_inputs.push_back(std::move(input_tensor_7));
//     ort_inputs.push_back(std::move(lane_node_masks_input_tensor));
//     ort_inputs.push_back(std::move(f4_input_tensor));
//     ort_inputs.push_back(std::move(nbr_vehicle_masks_input_tensor));
//     ort_inputs.push_back(std::move(f6_input_tensor));
//     ort_inputs.push_back(std::move(nbr_ped_masks_input_tensor));
//     ort_inputs.push_back(std::move(f8_input_tensor));
//     ort_inputs.push_back(std::move(f9_input_tensor));
//     ort_inputs.push_back(std::move(input_tensor_9));
//     ort_inputs.push_back(std::move(edge_type_input_tensor));

//     // score model & input tensor, get back output tensor
//     double timeStart = (double)getTickCount();
//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), 
//                                     ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 1);
//     double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
//     cout << "running time : " << nTime << " sec\n" << endl;

//     uint8_t m_numOutputs;
//     m_numOutputs = session.GetOutputCount();
//     printf("Model number of outputs: %d\n", m_numOutputs);

//     Ort::TypeInfo typeInfo = session.GetOutputTypeInfo(0);
//     auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

//     std::vector<int64_t> m_outputShapes = tensorInfo.GetShape();
//     for (int k = 0; k < m_outputShapes.size(); k++)
//     {
//         std::cout << m_outputShapes[k] << " ";
//     }
    
//     // Get pointer to output tensor float values
//     float* floatarr = output_tensors[0].GetTensorMutableData<float>();
//     // std::cout << floatarr[0] << std::endl;

//     printf("Done!\n");
//     return 0;
// }

