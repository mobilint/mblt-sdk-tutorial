#include "depth_postprocess.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

cv::Size resolve_depth_size(const std::vector<int64_t>& shape) {
    if (shape.size() == 2) {
        return cv::Size(static_cast<int>(shape[1]), static_cast<int>(shape[0]));
    }
    if (shape.size() == 3 && shape[0] == 1) {
        return cv::Size(static_cast<int>(shape[2]), static_cast<int>(shape[1]));
    }
    if (shape.size() == 3 && shape[2] == 1) {
        return cv::Size(static_cast<int>(shape[1]), static_cast<int>(shape[0]));
    }
    if (shape.size() == 4 && shape[0] == 1 && shape[1] == 1) {
        return cv::Size(static_cast<int>(shape[3]), static_cast<int>(shape[2]));
    }
    if (shape.size() == 4 && shape[0] == 1 && shape[3] == 1) {
        return cv::Size(static_cast<int>(shape[2]), static_cast<int>(shape[1]));
    }
    throw std::invalid_argument("Expected a single-channel depth output.");
}

float percentile(const std::vector<float>& sorted, float fraction) {
    const float position = fraction * static_cast<float>(sorted.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    const float weight = position - static_cast<float>(lower);
    return sorted[lower] * (1.0f - weight) + sorted[upper] * weight;
}

}  // namespace

cv::Mat postprocess_depth(const mobilint::NDArray<float>& output,
                          const LetterboxInfo& letterbox,
                          cv::Size original_size) {
    const cv::Size raw_size = resolve_depth_size(output.shape());
    if (raw_size.width * 4 != letterbox.input_width ||
        raw_size.height * 4 != letterbox.input_height) {
        throw std::invalid_argument(
            "The 4x MXQ depth output does not match the ONNX output shape.");
    }

    cv::Mat raw(raw_size.height, raw_size.width, CV_32FC1,
                const_cast<float*>(output.data()));

    // OpenCV's linear resize uses half-pixel sampling, matching PyTorch
    // bilinear interpolation with align_corners=false.
    cv::Mat onnx_depth;
    cv::resize(raw, onnx_depth,
               cv::Size(letterbox.input_width, letterbox.input_height),
               0, 0, cv::INTER_LINEAR);

    const int cropped_width =
        letterbox.input_width - letterbox.left - letterbox.right;
    const int cropped_height =
        letterbox.input_height - letterbox.top - letterbox.bottom;
    if (cropped_width <= 0 || cropped_height <= 0) {
        throw std::invalid_argument("Removing letterbox padding produced an empty depth map.");
    }

    const cv::Rect content(letterbox.left, letterbox.top,
                           cropped_width, cropped_height);
    cv::Mat restored;
    cv::resize(onnx_depth(content), restored, original_size,
               0, 0, cv::INTER_LINEAR);
    return restored;
}

cv::Mat visualize_depth(const cv::Mat& image_bgr,
                        const cv::Mat& depth,
                        double alpha) {
    if (image_bgr.empty() || depth.empty()) {
        throw std::invalid_argument("Image and depth map must not be empty.");
    }
    if (image_bgr.size() != depth.size()) {
        throw std::invalid_argument("Image and depth-map sizes must match.");
    }
    if (alpha < 0.0 || alpha > 1.0) {
        throw std::invalid_argument("Overlay alpha must be between 0 and 1.");
    }

    cv::Mat valid(depth.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat disparity(depth.size(), CV_32FC1, cv::Scalar(0));
    std::vector<float> valid_values;
    valid_values.reserve(depth.total());
    for (int row = 0; row < depth.rows; ++row) {
        const float* depth_row = depth.ptr<float>(row);
        float* disparity_row = disparity.ptr<float>(row);
        uint8_t* valid_row = valid.ptr<uint8_t>(row);
        for (int column = 0; column < depth.cols; ++column) {
            const float value = depth_row[column];
            if (std::isfinite(value) && value > 0.0f) {
                disparity_row[column] = 1.0f / value;
                valid_row[column] = 255;
                valid_values.push_back(disparity_row[column]);
            }
        }
    }
    if (valid_values.empty()) {
        throw std::invalid_argument("Depth output contains no positive finite values.");
    }

    std::sort(valid_values.begin(), valid_values.end());
    const float lower = percentile(valid_values, 0.02f);
    float upper = percentile(valid_values, 0.98f);
    if (upper <= lower) upper = lower + 1e-6f;

    cv::Mat normalized(depth.size(), CV_8UC1, cv::Scalar(0));
    for (int row = 0; row < depth.rows; ++row) {
        const float* disparity_row = disparity.ptr<float>(row);
        const uint8_t* valid_row = valid.ptr<uint8_t>(row);
        uint8_t* normalized_row = normalized.ptr<uint8_t>(row);
        for (int column = 0; column < depth.cols; ++column) {
            if (valid_row[column]) {
                const float scaled =
                    (disparity_row[column] - lower) * 255.0f / (upper - lower);
                normalized_row[column] =
                    cv::saturate_cast<uint8_t>(std::clamp(scaled, 0.0f, 255.0f));
            }
        }
    }

    cv::Mat colorized;
    cv::applyColorMap(normalized, colorized, cv::COLORMAP_JET);
    colorized.setTo(cv::Scalar(0, 0, 0), valid == 0);

    cv::Mat blended;
    cv::addWeighted(image_bgr, 1.0 - alpha, colorized, alpha, 0.0, blended);
    cv::Mat result = image_bgr.clone();
    blended.copyTo(result, valid);
    return result;
}
