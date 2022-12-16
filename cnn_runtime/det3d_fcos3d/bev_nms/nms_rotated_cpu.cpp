#include "nms_rotated_cpu.h"


void get_max_score_index(const std::vector<float> &scores,
                         const float threshold, const int top_k,
                         std::vector<std::pair<float, int>> *score_index_vec) {
  // Generate index score pairs.
  for (int i = 0; i < static_cast<int>(scores.size()); ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  // Sort the score pair according to the scores in descending order
  std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                   sort_score_pair_descend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < static_cast<int>(score_index_vec->size())) {
    score_index_vec->resize(top_k);
  }
}

void apply_nms_fast(const std::vector<RotatedBox<float>> &bboxes,
                    const std::vector<float> &scores,
                    const float score_threshold, const float nms_threshold,
                    const float eta, const int top_k,
                    std::vector<int> *indices) {
  // Sanity check.
//   CHECK_EQ(bboxes.size(), scores.size())
//       << "bboxes and scores have different size.";

  // Get top_k scores (with corresponding indices).
  std::vector<std::pair<float, int>> score_index_vec;
  get_max_score_index(scores, score_threshold, top_k, &score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices->clear();
  while (!score_index_vec.empty()) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    for (int k = 0; k < static_cast<int>(indices->size()); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        // bboxes[idx], bboxes[kept_idx]
        float bbox1[5] = {bboxes[idx].x_ctr, bboxes[idx].y_ctr, bboxes[idx].w, bboxes[idx].h, bboxes[idx].a};
        float bbox2[5] = {bboxes[kept_idx].x_ctr, bboxes[kept_idx].y_ctr, 
                        bboxes[kept_idx].w, bboxes[kept_idx].h, bboxes[kept_idx].a};
        float overlap = single_box_iou_rotated(bbox1, bbox2);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
    // if (keep && eta < 1 && adaptive_threshold > 0.5) {
    //   adaptive_threshold *= eta;
    // }
  }
}


// std::vector<std::vector<float>> YOLO::applyNMS(std::vector<std::vector<float>>& boxes,
// 	                                    const float thres) 
// {    
//     std::vector<std::vector<float>> result;
//     std::vector<bool> exist_box(boxes.size(), true);

//     int n = 0;
//     for (size_t _i = 0; _i < boxes.size(); ++_i) 
// 	{
//         if (!exist_box[_i]) 
// 			continue;
//         n = 0;
//         for (size_t _j = _i + 1; _j < boxes.size(); ++_j)
// 		{
//             // different class name
//             if (!exist_box[_j] || boxes[_i][4] != boxes[_j][4]) 
// 				continue;
//             float ovr = cal_iou(boxes[_j], boxes[_i]);
//             if (ovr >= thres) 
//             {
//                 if (boxes[_j][5] <= boxes[_i][5])
//                 {
//                     exist_box[_j] = false;
//                 }
//                 else
//                 {
//                     n++;   // have object_score bigger than boxes[_i]
//                     exist_box[_i] = false;
//                     break;
//                 }
//             }
//         }
//         //if (n) exist_box[_i] = false;
// 		if (n == 0) 
// 		{
// 			result.push_back(boxes[_i]);
// 		}			
//     }

//     return result;
// }

// template <typename scalar_t>
// at::Tensor nms_rotated_cpu_kernel(
//     const at::Tensor& dets,
//     const at::Tensor& scores,
//     const double iou_threshold) {
//   // nms_rotated_cpu_kernel is modified from torchvision's nms_cpu_kernel,
//   // however, the code in this function is much shorter because
//   // we delegate the IoU computation for rotated boxes to
//   // the single_box_iou_rotated function in box_iou_rotated_utils.h
//   AT_ASSERTM(dets.device().is_cpu(), "dets must be a CPU tensor");
//   AT_ASSERTM(scores.device().is_cpu(), "scores must be a CPU tensor");
//   AT_ASSERTM(
//       dets.scalar_type() == scores.scalar_type(),
//       "dets should have the same type as scores");

//   if (dets.numel() == 0) {
//     return at::empty({0}, dets.options().dtype(at::kLong));
//   }

//   auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

//   auto ndets = dets.size(0);
//   at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte));
//   at::Tensor keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));

//   auto suppressed = suppressed_t.data_ptr<uint8_t>();
//   auto keep = keep_t.data_ptr<int64_t>();
//   auto order = order_t.data_ptr<int64_t>();

//   int64_t num_to_keep = 0;

//   for (int64_t _i = 0; _i < ndets; _i++) {
//     auto i = order[_i];
//     if (suppressed[i] == 1) {
//       continue;
//     }

//     keep[num_to_keep++] = i;

//     for (int64_t _j = _i + 1; _j < ndets; _j++) {
//       auto j = order[_j];
//       if (suppressed[j] == 1) {
//         continue;
//       }

//       auto ovr = single_box_iou_rotated<scalar_t>(
//           dets[i].data_ptr<scalar_t>(), dets[j].data_ptr<scalar_t>());
//       if (ovr >= iou_threshold) {
//         suppressed[j] = 1;
//       }
//     }
//   }
//   return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep);
// }