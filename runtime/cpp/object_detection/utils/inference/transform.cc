// =============================================================================
// transform.cc - 이미지 변환기 구현 (Transformer implementation)
// =============================================================================
#include "transform.h"

#include <cstring>
#include <stdexcept>

int Transformer::parse_interpolation(const std::string& s) {
    if (s == "nearest") return cv::INTER_NEAREST;
    if (s == "bicubic") return cv::INTER_CUBIC;
    if (s == "area") return cv::INTER_AREA;
    return cv::INTER_LINEAR;
}

void Transformer::resize(cv::Mat& img, cv::Size size, const std::string& interpolation) {
    cv::resize(img, img, size, 0, 0, parse_interpolation(interpolation));
}

void Transformer::resize_short_edge(cv::Mat& img, int short_edge,
                                    const std::string& interpolation) {
    int h = img.rows;
    int w = img.cols;
    int min_hw = std::min(h, w);
    if (min_hw == short_edge) return;
    float scale = static_cast<float>(short_edge) / static_cast<float>(min_hw);
    int new_h = std::max(1, static_cast<int>(std::round(h * scale)));
    int new_w = std::max(1, static_cast<int>(std::round(w * scale)));
    cv::resize(img, img, cv::Size(new_w, new_h), 0, 0, parse_interpolation(interpolation));
}

void Transformer::center_crop(cv::Mat& img, cv::Size size) {
    int crop_w = std::min(size.width, img.cols);
    int crop_h = std::min(size.height, img.rows);
    int x = std::max(0, (img.cols - crop_w) / 2);
    int y = std::max(0, (img.rows - crop_h) / 2);
    img = img(cv::Rect(x, y, crop_w, crop_h)).clone();
}

void Transformer::letter_box(cv::Mat& img, cv::Size size) {
    int h = img.rows, w = img.cols;
    float ratio = std::min(static_cast<float>(size.height) / h,
                           static_cast<float>(size.width) / w);
    int new_w = static_cast<int>(std::round(w * ratio));
    int new_h = static_cast<int>(std::round(h * ratio));
    if (h != new_h || w != new_w) {
        cv::resize(img, img, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    }
    int dw = size.width - new_w;
    int dh = size.height - new_h;
    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;
    cv::copyMakeBorder(img, img, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
}

void Transformer::normalize(cv::Mat& img, const std::string& style) {
    if (img.depth() != CV_32F) img.convertTo(img, CV_32F);
    if (style == "torch") {
        const cv::Scalar mean_bgr(0.406, 0.456, 0.485);
        const cv::Scalar std_bgr(0.225, 0.224, 0.229);
        img *= (1.0f / 255.0f);
        cv::subtract(img, mean_bgr, img);
        cv::divide(img, std_bgr, img);
    } else if (style == "tf") {
        img.convertTo(img, CV_32F, 1.0 / 127.5, -1.0);
    } else if (style == "div255") {
        img *= (1.0f / 255.0f);
    } else {
        throw std::runtime_error("Unknown normalize style: " + style);
    }
}

std::unique_ptr<float[]> Transformer::operator()(const cv::Mat& input,
                                                 const ModelInfo& cfg) {
    cv::Mat img = input.clone();
    for (const auto& p : cfg.m_preprocess_list) {
        switch (p.op) {
        case PreProcessOps::RESIZE: {
            if (std::holds_alternative<std::pair<int, int>>(p.img_size)) {
                auto [h, w] = std::get<std::pair<int, int>>(p.img_size);
                resize(img, cv::Size(w, h), p.style);
            } else if (std::holds_alternative<int>(p.img_size)) {
                int s = std::get<int>(p.img_size);
                resize_short_edge(img, s, p.style);
            }
        } break;
        case PreProcessOps::CENTERCROP: {
            if (std::holds_alternative<std::pair<int, int>>(p.img_size)) {
                auto [h, w] = std::get<std::pair<int, int>>(p.img_size);
                center_crop(img, cv::Size(w, h));
            } else if (std::holds_alternative<int>(p.img_size)) {
                int s = std::get<int>(p.img_size);
                center_crop(img, cv::Size(s, s));
            }
        } break;
        case PreProcessOps::NORMALIZE: {
            normalize(img, p.style);
        } break;
        case PreProcessOps::YOLO: {
            if (std::holds_alternative<std::pair<int, int>>(p.img_size)) {
                auto [h, w] = std::get<std::pair<int, int>>(p.img_size);
                letter_box(img, cv::Size(w, h));
            }
        } break;
        }
    }

    if (img.depth() != CV_32F) img.convertTo(img, CV_32F);
    int h = img.rows, w = img.cols, c = img.channels();
    CV_Assert(img.isContinuous());

    const float* src = img.ptr<float>(0);
    auto out = std::make_unique<float[]>(h * w * c);
    float* dst = out.get();
    for (int i = 0; i < h * w; ++i) {
        for (int j = 0; j < c; ++j) {
            dst[c * i + j] = src[c * i + (2 - j)];  // BGR -> RGB
        }
    }
    return out;
}

std::unique_ptr<uint8_t[]> Transformer::transform_uint8(const cv::Mat& input,
                                                        const ModelInfo& cfg) {
    cv::Mat img = input.clone();
    for (const auto& p : cfg.m_preprocess_list) {
        if (p.op == PreProcessOps::YOLO) {
            if (std::holds_alternative<std::pair<int, int>>(p.img_size)) {
                auto [h, w] = std::get<std::pair<int, int>>(p.img_size);
                letter_box(img, cv::Size(w, h));
            }
        }
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    int h = img.rows, w = img.cols, c = img.channels();
    CV_Assert(img.isContinuous());

    auto out = std::make_unique<uint8_t[]>(h * w * c);
    const uint8_t* src = img.data;
    for (int ch = 0; ch < c; ++ch) {
        for (int i = 0; i < h * w; ++i) {
            out[ch * h * w + i] = src[i * c + ch];
        }
    }
    return out;
}
