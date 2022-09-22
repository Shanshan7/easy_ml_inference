#include <vector>
#include <stdlib.h>
#include <iostream>

#include "onnxruntime_cxx_api.h"
#include "tensorrt_provider_factory.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include "JoysonPredictionCommon.h"


using namespace cv;
using namespace std;

/**
 * @file    ImageRecognitionOrtSessionHandlerBase.cpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#include <cassert>
#include <cstring>
#include <sstream>

#include "pgp_net_infer.h"

namespace Ort
{
PgpNetInference::PgpNetInference(
    const std::string& modelPath,  //
    const std::optional<size_t>& gpuIdx, const std::optional<std::vector<std::vector<int64_t>>>& inputShapes)
    : OrtSessionHandler(modelPath, gpuIdx, inputShapes)
{

}

PgpNetInference::~PgpNetInference()
{
}
}  // namespace Ort
