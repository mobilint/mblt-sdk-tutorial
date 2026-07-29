#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/opencv.hpp>

struct LetterboxInfo {
    int input_height;
    int input_width;
    int top;
    int bottom;
    int left;
    int right;
};

struct PreprocessedImage {
    std::vector<uint8_t> data;
    LetterboxInfo letterbox;
};

class Preprocessor {
public:
    // Applies 114-padded YOLO letterboxing, converts BGR to RGB, and packs the
    // uint8 pixels in the layout reported by the MXQ model.
    static PreprocessedImage transform_uint8(const cv::Mat& input,
                                             int input_height,
                                             int input_width,
                                             bool channel_last);
};
