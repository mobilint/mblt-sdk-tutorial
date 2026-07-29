#include "semantic_postprocess.h"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kNumCityscapesClasses = 19;

struct LogitsLayout {
    int height;
    int width;
    bool channel_last;
};

LogitsLayout resolve_logits_layout(const std::vector<int64_t>& shape) {
    if (shape.size() == 3 && shape[2] == kNumCityscapesClasses) {
        return {static_cast<int>(shape[0]), static_cast<int>(shape[1]), true};
    }
    if (shape.size() == 3 && shape[0] == kNumCityscapesClasses) {
        return {static_cast<int>(shape[1]), static_cast<int>(shape[2]), false};
    }
    if (shape.size() == 4 && shape[0] == 1 &&
        shape[3] == kNumCityscapesClasses) {
        return {static_cast<int>(shape[1]), static_cast<int>(shape[2]), true};
    }
    if (shape.size() == 4 && shape[0] == 1 &&
        shape[1] == kNumCityscapesClasses) {
        return {static_cast<int>(shape[2]), static_cast<int>(shape[3]), false};
    }
    throw std::invalid_argument(
        "Expected HWC or CHW logits with 19 Cityscapes classes.");
}

const std::array<cv::Vec3b, kNumCityscapesClasses> kCityscapesPaletteBgr = {
    cv::Vec3b{128, 64, 128},  // road
    cv::Vec3b{232, 35, 244},  // sidewalk
    cv::Vec3b{70, 70, 70},    // building
    cv::Vec3b{156, 102, 102}, // wall
    cv::Vec3b{153, 153, 190}, // fence
    cv::Vec3b{153, 153, 153}, // pole
    cv::Vec3b{30, 170, 250},  // traffic light
    cv::Vec3b{0, 220, 220},   // traffic sign
    cv::Vec3b{35, 142, 107},  // vegetation
    cv::Vec3b{152, 251, 152}, // terrain
    cv::Vec3b{180, 130, 70},  // sky
    cv::Vec3b{60, 20, 220},   // person
    cv::Vec3b{0, 0, 255},     // rider
    cv::Vec3b{142, 0, 0},     // car
    cv::Vec3b{70, 0, 0},      // truck
    cv::Vec3b{100, 60, 0},    // bus
    cv::Vec3b{100, 80, 0},    // train
    cv::Vec3b{230, 0, 0},     // motorcycle
    cv::Vec3b{32, 11, 119},   // bicycle
};

}  // namespace

cv::Mat postprocess_semantic(const mobilint::NDArray<float>& output,
                             const LetterboxInfo& letterbox,
                             cv::Size original_size) {
    const LogitsLayout layout = resolve_logits_layout(output.shape());
    if (layout.height != letterbox.input_height ||
        layout.width != letterbox.input_width) {
        throw std::invalid_argument(
            "MXQ and ONNX semantic output spatial shapes do not match.");
    }

    const float* logits = output.data();
    const int pixels = layout.height * layout.width;
    cv::Mat class_map(layout.height, layout.width, CV_8UC1);
    for (int pixel = 0; pixel < pixels; ++pixel) {
        float best_value = -std::numeric_limits<float>::infinity();
        int best_class = 0;
        for (int class_id = 0; class_id < kNumCityscapesClasses;
             ++class_id) {
            const int index =
                layout.channel_last
                    ? pixel * kNumCityscapesClasses + class_id
                    : class_id * pixels + pixel;
            if (logits[index] > best_value) {
                best_value = logits[index];
                best_class = class_id;
            }
        }
        class_map.data[pixel] = static_cast<uint8_t>(best_class);
    }

    const int cropped_width =
        letterbox.input_width - letterbox.left - letterbox.right;
    const int cropped_height =
        letterbox.input_height - letterbox.top - letterbox.bottom;
    if (cropped_width <= 0 || cropped_height <= 0) {
        throw std::invalid_argument(
            "Removing letterbox padding produced an empty class map.");
    }

    const cv::Rect content(letterbox.left, letterbox.top,
                           cropped_width, cropped_height);
    if (content.size() == original_size) {
        return class_map(content).clone();
    }
    cv::Mat restored;
    cv::resize(class_map(content), restored, original_size,
               0, 0, cv::INTER_NEAREST);
    return restored;
}

cv::Mat visualize_semantic(const cv::Mat& image_bgr,
                           const cv::Mat& class_map,
                           double alpha) {
    if (image_bgr.empty() || class_map.empty()) {
        throw std::invalid_argument(
            "Image and semantic class map must not be empty.");
    }
    if (image_bgr.size() != class_map.size()) {
        throw std::invalid_argument(
            "Image and semantic class-map sizes must match.");
    }
    if (class_map.type() != CV_8UC1) {
        throw std::invalid_argument(
            "Semantic class map must contain uint8 class IDs.");
    }
    if (alpha < 0.0 || alpha > 1.0) {
        throw std::invalid_argument(
            "Overlay alpha must be between 0 and 1.");
    }

    cv::Mat overlay(image_bgr.size(), CV_8UC3);
    for (int row = 0; row < class_map.rows; ++row) {
        const uint8_t* class_row = class_map.ptr<uint8_t>(row);
        cv::Vec3b* overlay_row = overlay.ptr<cv::Vec3b>(row);
        for (int column = 0; column < class_map.cols; ++column) {
            const uint8_t class_id = class_row[column];
            if (class_id >= kCityscapesPaletteBgr.size()) {
                throw std::invalid_argument(
                    "Cityscapes class ID must be between 0 and 18.");
            }
            overlay_row[column] = kCityscapesPaletteBgr[class_id];
        }
    }

    cv::Mat result;
    cv::addWeighted(image_bgr, 1.0 - alpha, overlay, alpha, 0.0, result);
    return result;
}
