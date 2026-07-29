#pragma once

#include <opencv2/opencv.hpp>
#include <qbruntime/qbruntime.h>

#include "preprocessor.h"

// Applies argmax to 19-class HWC or CHW logits, removes letterbox padding,
// and restores the source-image shape.
cv::Mat postprocess_semantic(const mobilint::NDArray<float>& output,
                             const LetterboxInfo& letterbox,
                             cv::Size original_size);

// Applies the official Cityscapes palette and blends it over the source image.
cv::Mat visualize_semantic(const cv::Mat& image_bgr,
                           const cv::Mat& class_map,
                           double alpha = 0.7);
