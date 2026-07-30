// Image-to-NPU-tensor preprocessor: applies ModelInfo.m_preprocess_list ops in order.
#pragma once
#include <memory>
#include <opencv2/opencv.hpp>

#include "types.h"

class Preprocessor {
 public:
  // Float HWC RGB buffer from the full preprocess pipeline (resize/crop/normalize/letterbox).
  std::unique_ptr<float[]> operator()(const cv::Mat& input, const ModelInfo& cfg);

  // uint8 HWC RGB buffer (letterbox only) for uint8-input MXQ models. Feed via Model::infer.
  std::unique_ptr<uint8_t[]> transform_uint8(const cv::Mat& input, const ModelInfo& cfg);

  // Float HWC RGB buffer normalized to /255 for float-input (!uint8) MXQ models;
  // same layout as transform_uint8. Feed via Model::infer.
  std::unique_ptr<float[]> transform_float(const cv::Mat& input, const ModelInfo& cfg);

  // Channel-first (CHW) counterparts, used only when getModelInputShape() is channel-first.
  // Same pixels as the HWC versions, reordered to CHW; feed via Model::inferCHW.
  std::unique_ptr<uint8_t[]> transform_uint8_chw(const cv::Mat& input, const ModelInfo& cfg);
  std::unique_ptr<float[]> transform_float_chw(const cv::Mat& input, const ModelInfo& cfg);

 private:
  void resize(cv::Mat& img, cv::Size size, const std::string& interpolation);
  void resize_short_edge(cv::Mat& img, int short_edge, const std::string& interpolation);
  void center_crop(cv::Mat& img, cv::Size size);
  void normalize(cv::Mat& img, const std::string& style);
  void letter_box(cv::Mat& img, cv::Size size);
  int parse_interpolation(const std::string& s);
};
