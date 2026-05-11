// =============================================================================
// transform.h - 이미지 -> NPU 입력 텐서 변환
// =============================================================================
// YAML config (ModelInfo) 의 m_preprocess_list 를 순서대로 적용.
// Applies m_preprocess_list from ModelInfo (parsed from YAML config) in order.
// =============================================================================
#pragma once
#include <memory>
#include <opencv2/opencv.hpp>

#include "parser.h"

class Transformer {
public:
    // float HWC RGB 출력 (NPU float-input MXQ 용)
    std::unique_ptr<float[]> operator()(const cv::Mat& input, const ModelInfo& cfg);

    // uint8 CHW RGB 출력 (NPU uint8-input MXQ 용, 튜토리얼과 동일)
    std::unique_ptr<uint8_t[]> transform_uint8(const cv::Mat& input, const ModelInfo& cfg);

private:
    void resize(cv::Mat& img, cv::Size size, const std::string& interpolation);
    void resize_short_edge(cv::Mat& img, int short_edge, const std::string& interpolation);
    void center_crop(cv::Mat& img, cv::Size size);
    void normalize(cv::Mat& img, const std::string& style);
    void letter_box(cv::Mat& img, cv::Size size);
    int parse_interpolation(const std::string& s);
};
