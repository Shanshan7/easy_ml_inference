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
#include <iostream>
#include <vector>

#include "nms_rotated_cpu.h"

// example
// origin:  tensor([[ 0.9164, 50.0849,  5.8645, 52.1092, -4.7290],
//         [ 0.6789, 20.6082,  5.3573, 22.5448, -4.6963],
//         [-2.6247, 18.9857,  1.9290, 20.9115, -1.5794],
//         [-2.6207, 19.1425,  1.9366, 21.0649, -1.5749],
//         [-2.6314, 19.0022,  1.9536, 20.9317, -1.5845],
//         [-2.6277, 19.0674,  1.9360, 20.9924, -1.5821],
//         [ 0.6619, 20.6629,  5.3714, 22.5860, -4.6964],
//         [ 0.6627, 20.7021,  5.3484, 22.6164, -4.7084],
//         [ 0.6715, 20.8196,  5.3700, 22.7297, -4.6680],
//         [ 0.6705, 20.5443,  5.3598, 22.4592, -4.6763]], device='cuda:0')
// scores:  tensor([0.2264, 0.4245, 0.3434, 0.2965, 0.2663, 0.2640, 0.2406, 0.2325, 0.2101,
//         0.2048], device='cuda:0')
// selected:  tensor([[ 0.6789, 20.6082,  5.3573, 22.5448, -4.6963],
//         [-2.6247, 18.9857,  1.9290, 20.9115, -1.5794],
//         [ 0.9164, 50.0849,  5.8645, 52.1092, -4.7290]], device='cuda:0')
// origin:  tensor([[ 9.0052, 25.0572,  9.6689, 25.7017, -4.6698],
//         [ 9.0348, 25.1043,  9.7255, 25.7650, -4.6745],
//         [ 9.9411, 50.3440, 10.6551, 51.0240, -1.5913],
//         [10.1696, 49.1633, 10.8819, 49.8386, -1.5950]], device='cuda:0')
// scores:  tensor([0.3994, 0.2991, 0.2279, 0.2187], device='cuda:0')
// selected:  tensor([[ 9.0052, 25.0572,  9.6689, 25.7017, -4.6698],
//         [ 9.9411, 50.3440, 10.6551, 51.0240, -1.5913],
//         [10.1696, 49.1633, 10.8819, 49.8386, -1.5950]], device='cuda:0')

int main()
{
    std::vector<RotatedBox<float>> origin_boxes_vector;
    std::vector<float> scores = {0.2264, 0.4245, 0.3434, 0.2965, 0.2663, 0.2640, 0.2406, 0.2325, 0.2101, 0.2048};
    RotatedBox<float> origin_box, selected_box;

    // example first
    // origin_box.x_ctr = (9.0052 + 9.6689) / 2;
    // origin_box.y_ctr = (25.0572 + 25.7017) / 2;
    // origin_box.w = 9.6689 - 9.0052;
    // origin_box.h = 25.7017 - 25.0572;
    // origin_box.a = -4.6698;
    // origin_boxes_vector.push_back(origin_box);

    // origin_box.x_ctr = (9.0348 + 9.7255) / 2;
    // origin_box.y_ctr = (25.1043 + 25.7650) / 2;
    // origin_box.w = 9.7255 - 9.0348;
    // origin_box.h = 25.7650 - 25.1043;
    // origin_box.a = -4.6745;
    // origin_boxes_vector.push_back(origin_box);

    // origin_box.x_ctr = (9.9411 + 10.6551) / 2;
    // origin_box.y_ctr = (50.3440 + 51.0240) / 2;
    // origin_box.w = 10.6551 - 9.9411;
    // origin_box.h = 51.0240 - 50.3440;
    // origin_box.a = -1.5913;
    // origin_boxes_vector.push_back(origin_box);

    // origin_box.x_ctr = (10.1696 + 10.8819) / 2;
    // origin_box.y_ctr = (49.1633 + 49.8386) / 2;
    // origin_box.w = 10.8819 - 10.1696;
    // origin_box.h = 49.8386 - 49.1633;
    // origin_box.a = -1.5950;
    // origin_boxes_vector.push_back(origin_box);

    // example second
    origin_box.x_ctr = (0.9164 + 5.8645) / 2;
    origin_box.y_ctr = (50.0849 + 52.1092) / 2;
    origin_box.w = 5.8645 - 0.9164;
    origin_box.h = 52.1092 - 50.0849;
    origin_box.a = -4.7290;
    origin_boxes_vector.push_back(origin_box);

    origin_box.x_ctr = (0.6789 + 5.3573) / 2;
    origin_box.y_ctr = (20.6082 + 22.5448) / 2;
    origin_box.w = 5.3573 - 0.6789;
    origin_box.h = 22.5448 - 20.6082;
    origin_box.a = -4.6963;
    origin_boxes_vector.push_back(origin_box);

    origin_box.x_ctr = (-2.6247 + 1.9290) / 2;
    origin_box.y_ctr = (18.9857 + 20.9115) / 2;
    origin_box.w = 1.9290 + 2.6247;
    origin_box.h = 20.9115 - 18.9857;
    origin_box.a = -1.5794;
    origin_boxes_vector.push_back(origin_box);

    origin_box.x_ctr = (-2.6207 + 1.9366) / 2;
    origin_box.y_ctr = (19.1425 + 21.0649) / 2;
    origin_box.w = 1.9366 + 2.6207;
    origin_box.h = 21.0649 - 19.1425;
    origin_box.a = -1.5749;
    origin_boxes_vector.push_back(origin_box);

    origin_box.x_ctr = (-2.6314 + 1.9536) / 2;
    origin_box.y_ctr = (19.0022 + 20.9317) / 2;
    origin_box.w = 1.9536 + 2.6314;
    origin_box.h = 20.9317 - 19.0022;
    origin_box.a = -1.5845;
    origin_boxes_vector.push_back(origin_box);

    origin_box.x_ctr = (-2.6277 + 1.9360) / 2;
    origin_box.y_ctr = (19.0674 + 20.9924) / 2;
    origin_box.w = 1.9360 + 2.6277;
    origin_box.h = 20.9924 - 19.0674;
    origin_box.a = -1.5821;
    origin_boxes_vector.push_back(origin_box);

    origin_box.x_ctr = (0.6619 + 5.3714) / 2;
    origin_box.y_ctr = (20.6629 + 22.5860) / 2;
    origin_box.w = 5.3714 - 0.6619;
    origin_box.h = 22.5860 - 20.6629;
    origin_box.a = -4.6964;
    origin_boxes_vector.push_back(origin_box);

    origin_box.x_ctr = (0.6627 + 5.3484) / 2;
    origin_box.y_ctr = (20.7021 + 22.6164) / 2;
    origin_box.w = 5.3484 - 0.6627;
    origin_box.h = 22.6164 - 20.7021;
    origin_box.a = -4.7084;
    origin_boxes_vector.push_back(origin_box);

    origin_box.x_ctr = (0.6715 + 5.3700) / 2;
    origin_box.y_ctr = (20.8196 + 22.7297) / 2;
    origin_box.w = 5.3700 - 0.6715;
    origin_box.h = 22.7297 - 20.8196;
    origin_box.a = -4.6680;
    origin_boxes_vector.push_back(origin_box);

    origin_box.x_ctr = (0.6705 + 5.3598) / 2;
    origin_box.y_ctr = (20.5443 + 22.4592) / 2;
    origin_box.w = 5.3598 - 0.6705;
    origin_box.h = 22.4592 - 20.5443;
    origin_box.a = -4.6763;
    origin_boxes_vector.push_back(origin_box);

    std::vector<int> indices;
    apply_nms_fast(origin_boxes_vector, scores, 0.1, 0.45, 0.01, origin_boxes_vector.size(), &indices);

    std::cout << "len: " << indices.size() << std::endl;

    for(int i = 0; i < indices.size(); i++)
    {
        std::cout << i << " " << indices[i] << std::endl;
    }

    std::vector<int> labels;
    labels.resize(10);

    std::vector<std::vector<int>> labels_vec;
    for(int k = 0; k < 5; k++)
    {
        labels_vec.push_back(labels);
    }
    std::cout << labels_vec.size() << " " << labels_vec[0].size() << std::endl;
    int vector_len = sizeof(labels) / sizeof(int);
    std::cout << "vector_len: " << vector_len << std::endl;

    return 0;
    
}

