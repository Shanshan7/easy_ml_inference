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
#pragma once

#include <vector>

#include "box_iou_rotated_utils.h"

template <typename T>
bool sort_score_pair_descend(const std::pair<float, T> &pair1,
                             const std::pair<float, T> &pair2) {
  return pair1.first > pair2.first;
}

void get_max_score_index(const std::vector<float> &scores,
                         const float threshold, const int top_k,
                         std::vector<std::pair<float, int>> *score_index_vec);

void apply_nms_fast(const std::vector<RotatedBox<float>> &bboxes,
                    const std::vector<float> &scores,
                    const float score_threshold, const float nms_threshold,
                    const float eta, const int top_k,
                    std::vector<int> *indices);