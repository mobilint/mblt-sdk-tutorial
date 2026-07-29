// YOLO anchorless instance-segmentation post-processor: converts raw NPU output tensors into
// (x1, y1, x2, y2, conf, cls) detections plus per-detection mask coefficients, then assembles
// instance masks from the prototype tensor.
// Targets YOLOv8/v11 anchor-free Segment heads only (not anchor-based variants).
//
// LAYOUT: assumes all NPU output tensors are HWC (channel-last), including the prototype
// ([proto_h, proto_w, num_mask_coeffs]): value(spatial, channel) is at base[spatial*channels + channel].
// This matches Model::infer (HWC). Feeding Model::inferCHW output (CHW) would mis-index channels.
//
// Algorithm:
//   1) Split raw outputs by stride into box (reg_max*4 ch), cls (nc ch), and mask-coeff (num_mask_coeffs ch)
//      tensors, plus one prototype tensor [num_mask_coeffs, proto_h, proto_w] (the largest num_mask_coeffs map).
//   2) Flatten anchors across all strides -> total_anchors grid points.
//   3) Pre-filter by max cls logit > invsigmoid(conf_thres) to skip most anchors cheaply.
//   4) DFL softmax + expectation -> 4-side distances -> bbox (xyxy, letterbox coordinates).
//   5) sigmoid(cls) then filter conf > conf_thres; carry the matching mask coefficients.
//   6) Apply per-class coordinate offset then run NMS.
//   7) mask = sigmoid(coeff @ prototypes) -> upsample to img_size -> crop to box -> threshold > 0.0.
//   8) Strip letterbox padding then resize masks to original image and threshold > 0.5.
#pragma once
#include <qbruntime/ndarray.h>

#include <opencv2/opencv.hpp>
#include <vector>

class YoloSegDecoder {
 public:
  struct Detection {
    float x1, y1, x2, y2, conf;
    int cls;
    std::vector<float> mask_coeffs;  // num_mask_coeffs mask coefficients
  };

  YoloSegDecoder(int nc, int nl, int img_size, int reg_max = 16, int num_mask_coeffs = 32, float conf_thres = 0.25f,
                 float iou_thres = 0.45f, int max_det = 300);

  // Decodes raw NPU output tensors (NDArray, HWC float32) into detections in letterbox coordinates.
  // Fills proto_out with the prototype tensor data and sets proto_c/proto_h/proto_w to its dimensions.
  std::vector<Detection> decode(const std::vector<mobilint::NDArray<float>>& raw_outputs, std::vector<float>& proto_out,
                                int& proto_c, int& proto_h, int& proto_w) const;

  // Assembles one binary instance mask per detection at the original image resolution:
  // combines mask coefficients with the prototype tensor, upsamples to letterbox space,
  // crops to each box, strips letterbox padding, and resizes to (orig_h, orig_w).
  // Returns one CV_8U mask (0/255) of size (orig_h, orig_w) per detection.
  std::vector<cv::Mat> assemble_masks(const std::vector<Detection>& dets, const std::vector<float>& proto, int proto_c,
                                      int proto_h, int proto_w, int orig_h, int orig_w) const;

  // Rescales detection boxes from letterbox (img_size x img_size) space to original image coordinates.
  static void scale_to_original(std::vector<Detection>& dets, int img_size, int orig_h, int orig_w);

 private:
  int nc_;
  int nl_;
  int img_size_;
  int reg_max_;
  int num_mask_coeffs_;
  float conf_thres_;
  float iou_thres_;
  int max_det_;
  float invconf_;                                 // invsigmoid(conf_thres) for logit-space pre-filter
  std::vector<int> strides_;                      // per-stride values [8, 16, 32, ...]
  std::vector<int> grid_sizes_;                   // per-stride H*W grid cell counts
  std::vector<std::pair<float, float>> anchors_;  // flattened anchor centers (cx, cy) across all strides
  std::vector<float> stride_per_anchor_;          // stride value for each anchor entry
};
