#pragma once

#include <qbruntime/qbruntime.h>

#include <opencv2/opencv.hpp>

#include "preprocessor.h"

// Performs the C++ equivalent of:
// F.interpolate(depth, scale_factor=4.0, mode="bilinear", align_corners=False)
// and then removes letterbox padding and restores the source-image shape.
cv::Mat postprocess_depth(const mobilint::NDArray<float>& output, const LetterboxInfo& letterbox,
                          cv::Size original_size);

// Renders near regions in warm colors and far regions in cool colors.
cv::Mat visualize_depth(const cv::Mat& image_bgr, const cv::Mat& depth, double alpha = 0.7);
