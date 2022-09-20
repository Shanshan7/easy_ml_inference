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

#include "JoysonFusionCommon.h"
#include "JoysonPredictionPre.h"


int main(int argc, char** argv)
{
    int rval = 0;

    // data preprocess
    JoysonPredictionPre joyson_prediction_pre;
    PredictionNetInput prediction_net_input;
    rval = joyson_prediction_pre.GetInputs("./pickle.json", prediction_net_input);

    return rval;
}