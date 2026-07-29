// YOLO anchorless detection post-processor: converts raw NPU output tensors into
// (x1, y1, x2, y2, conf, cls) detections using DFL decode and class-offset NMS.
// Targets YOLOv8/v9/v11 anchor-free Detect heads only (not anchor-based variants).
//
// LAYOUT: assumes NPU output tensors are HWC (channel-last), i.e. value(spatial, channel)
// lives at base[spatial * num_channels + channel]. This matches Model::infer (HWC).
// Feeding Model::inferCHW output (CHW) here would mis-index channels and yield garbage.
//
// Algorithm:
//   1) Split raw outputs by stride into box (reg_max*4 channels) and cls (nc channels) tensors.
//   2) Flatten anchors across all strides -> total_anchors grid points.
//   3) Pre-filter by max cls logit > invsigmoid(conf_thres) to skip most anchors cheaply.
//   4) DFL softmax + expectation -> 4-side distances -> bbox (xyxy, letterbox coordinates).
//   5) sigmoid(cls) then filter conf > conf_thres.
//   6) Apply per-class coordinate offset then run NMS.
#pragma once
#include <qbruntime/ndarray.h>

#include <vector>

class YoloDecoder {
 public:
  struct Detection {
    float x1, y1, x2, y2, conf;
    int cls;
  };

  YoloDecoder(int nc, int nl, int img_size, int reg_max = 16, float conf_thres = 0.25f, float iou_thres = 0.7f,
              int max_det = 300);

  // Decodes raw NPU output tensors (NDArray, HWC float32) into detections in letterbox coordinates.
  std::vector<Detection> decode(const std::vector<mobilint::NDArray<float>>& raw_outputs) const;

  // Rescales detections from letterbox (img_size x img_size) space to original image coordinates.
  static void scale_to_original(std::vector<Detection>& dets, int img_size, int orig_h, int orig_w);

 private:
  int nc_;
  int nl_;
  int img_size_;
  int reg_max_;
  float conf_thres_;
  float iou_thres_;
  int max_det_;
  float invconf_;                                 // invsigmoid(conf_thres) for logit-space pre-filter
  std::vector<int> strides_;                      // per-stride values [8, 16, 32, ...]
  std::vector<int> grid_sizes_;                   // per-stride H*W grid cell counts
  std::vector<std::pair<float, float>> anchors_;  // flattened anchor centers (cx, cy) across all strides
  std::vector<float> stride_per_anchor_;          // stride value for each anchor entry
};
