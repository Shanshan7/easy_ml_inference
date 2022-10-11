/******************************************************************************
**                                                                           **
** Copyright (C) Joyson Electronics (2022)                                   **
**                                                                           **
** All rights reserved.                                                      **
**                                                                           **
** This document contains proprietary information belonging to Joyson        **
** Electronics. Passing on and copying of this document, and communication   **
** of its contents is not permitted without prior written authorization.     **
**                                                                           **
******************************************************************************/

// #include "JoysonFusionCommon.h"
#include "JoysonPredictionPre.h"

#include "pgp_net_infer.h"
#include "OrtSessionHandler.h"

int main(int argc, char** argv)
{
    int rval = 0;

    // data preprocess
    JoysonPredictionPre joyson_prediction_pre;
    PredictionNetInput prediction_net_input;
    std::vector<float*> inputImgData;
    inputImgData.resize(11);
    // inputImgData.clear();

    rval = joyson_prediction_pre.GetInputs("/docker_data/data/0a4c8e815185464db3c7304a41d47598_1c6da72ffe0f42088dc3090402fd718c.json", prediction_net_input);
    joyson_prediction_pre.NetInputTransform(prediction_net_input, inputImgData);

    std::string model_path = "./config/model/pgp.onnx";
    Ort::PgpNetInference osh(model_path, 0);

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            std::cout << inputImgData[0][i*5+j] << std::endl;
        }
    }
    auto inferenceOutput = osh(inputImgData);
    std::cout << inferenceOutput.size() << std::endl;

    return rval;
}