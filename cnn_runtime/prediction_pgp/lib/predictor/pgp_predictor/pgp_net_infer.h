/**
 * @file    ImageRecognitionOrtSessionHandlerBase.hpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "OrtSessionHandler.h"

namespace Ort
{
class PgpNetInference : public OrtSessionHandler
{
 public:
    PgpNetInference(                          //
        const std::string& modelPath,                        //
        const std::optional<size_t>& gpuIdx = std::nullopt,  //
        const std::optional<std::vector<std::vector<int64_t>>>& inputShapes = std::nullopt);

    ~PgpNetInference();

    // virtual void preprocess(float* dst,                              //
    //                         const unsigned char* src,                //
    //                         const int64_t targetImgWidth,            //
    //                         const int64_t targetImgHeight,           //
    //                         const int numChanels,                    //
    //                         const std::vector<float>& meanVal = {},  //
    //                         const std::vector<float>& stdVal = {}) const;

    std::vector<DataOutputType> outputData;
};
}  // namespace Ort
