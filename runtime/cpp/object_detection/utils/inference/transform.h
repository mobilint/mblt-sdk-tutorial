// Image-to-NPU-tensor transformer that applies m_preprocess_list ops from ModelInfo in order.
// operator() produces a float HWC->CHW RGB buffer for float-input MXQ models.
// transform_uint8() produces a uint8 CHW RGB buffer for uint8-input MXQ models.
//
// (KR) ModelInfo.m_preprocess_list 의 전처리 연산을 순서대로 적용해 NPU 입력 텐서를 생성하는 변환기.
// operator() 는 float-input MXQ 용 float CHW RGB 버퍼를 생성한다.
// transform_uint8() 는 uint8-input MXQ 용 uint8 CHW RGB 버퍼를 생성한다.
#pragma once
#include <memory>
#include <opencv2/opencv.hpp>

#include "types.h"

class Transformer {
public:
    // Produces a float CHW RGB buffer for float-input MXQ models. (KR: float-input MXQ 용 float CHW RGB 버퍼 생성.)
    std::unique_ptr<float[]> operator()(const cv::Mat& input, const ModelInfo& cfg);

    // Produces a uint8 CHW RGB buffer for uint8-input MXQ models. (KR: uint8-input MXQ 용 uint8 CHW RGB 버퍼 생성.)
    std::unique_ptr<uint8_t[]> transform_uint8(const cv::Mat& input, const ModelInfo& cfg);

private:
    void resize(cv::Mat& img, cv::Size size, const std::string& interpolation);
    void resize_short_edge(cv::Mat& img, int short_edge, const std::string& interpolation);
    void center_crop(cv::Mat& img, cv::Size size);
    void normalize(cv::Mat& img, const std::string& style);
    void letter_box(cv::Mat& img, cv::Size size);
    int parse_interpolation(const std::string& s);
};
