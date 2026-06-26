// End-to-end pose estimation inference on Mobilint NPU with skeleton visualization.
// Preprocessing (letterbox + BGR->RGB + HWC->CHW) is handled by Transformer.
// Normalization is fused into the MXQ model (uint8 input), so no float scaling is needed here.
// Pipeline: load MXQ -> transform uint8 -> NPU infer -> DFL decode -> keypoint decode -> NMS -> draw boxes + skeleton.
//
// Usage:
//   ./infer-pose <model.mxq> <image_path> <output_path>
//
// Examples:
//   ./infer-pose yolo11m-pose.mxq cr7.jpg result.jpg   # ARIES
//   ./infer-pose yolov8m-pose.mxq cr7.jpg result.jpg   # REGULUS
//
// (KR) Mobilint NPU 에서 포즈 추정 추론을 실행하고 스켈레톤을 이미지에 그린다.
// 전처리(letterbox + BGR->RGB + HWC->CHW)는 Transformer 가 담당한다.
// 정규화는 MXQ 모델에 퓨즈되어 있어(uint8 입력) 별도 float 변환이 필요 없다.
// 파이프라인: MXQ 로드 -> uint8 변환 -> NPU 추론 -> DFL 디코드 -> 키포인트 디코드 -> NMS -> 박스 + 스켈레톤 시각화.

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "decode.h"
#include "runner.h"
#include "transform.h"
#include "yolo_pose_config.h"

// COCO 17-keypoint skeleton as 1-indexed keypoint pairs (mirrors coco.py POSE_SKELETON).
// (KR) COCO 17 키포인트 스켈레톤, 1-기반 키포인트 쌍 (coco.py POSE_SKELETON 동일).
static const int POSE_SKELETON[19][2] = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
    {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
    {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7},
};

// Base 20-color pose palette in RGB (mirrors coco.py POSE_PALETTE).
// (KR) RGB 기준 20색 pose 팔레트 (coco.py POSE_PALETTE 동일).
static const cv::Scalar POSE_PALETTE[20] = {
    {255, 128, 0},   {255, 153, 51},  {255, 178, 102}, {230, 230, 0},
    {255, 153, 255}, {153, 204, 255}, {255, 102, 255}, {255, 51, 255},
    {102, 178, 255}, {51, 153, 255},  {255, 153, 153}, {255, 102, 102},
    {255, 51, 51},   {153, 255, 153}, {102, 255, 102}, {51, 255, 51},
    {0, 255, 0},     {0, 0, 255},     {255, 0, 0},     {255, 255, 255},
};

// Palette index per skeleton limb (mirrors coco.py LIMB_PALLETE). (KR) 스켈레톤 limb 별 팔레트 인덱스.
static const int LIMB_PALETTE_IDX[19] = {
    9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16};

// Palette index per keypoint (mirrors coco.py KEYPOINT_PALLETE). (KR) 키포인트 별 팔레트 인덱스.
static const int KEYPOINT_PALETTE_IDX[17] = {
    16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9};

// Keypoint visibility threshold for drawing (mirrors draw_kpts default conf=0.5 in visualize.py).
// (KR) 그리기용 키포인트 가시성 임계값 (visualize.py draw_kpts 기본 conf=0.5 동일).
static const float KPT_DRAW_THRES = 0.5f;

void draw_poses(cv::Mat& img,
                const std::vector<YoloPoseDecoder::Detection>& dets) {
    int h = img.rows;
    int w = img.cols;
    // Line/font thickness scales with image size (mirrors visualize.py draw_boxes). (KR: 선/폰트 두께는 이미지 크기에 비례.)
    int tl = static_cast<int>(std::round(0.002 * (h + w) / 2.0)) + 1;
    int tf = std::max(tl - 1, 1);
    int radius = 5;
    int limb_thickness = static_cast<int>(std::ceil(2.0 / 2.0));

    for (const auto& d : dets) {
        int x1 = static_cast<int>(d.x1);
        int y1 = static_cast<int>(d.y1);
        int x2 = static_cast<int>(d.x2);
        int y2 = static_cast<int>(d.y2);
        float pct = d.conf * 100.0f;

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                      cv::Scalar(0, 255, 0), tl, cv::LINE_AA);

        std::ostringstream label;
        label << "person: " << std::fixed << std::setprecision(1) << pct << "%";
        cv::putText(img, label.str(), cv::Point(x1, y1 - 2),
                    cv::FONT_HERSHEY_SIMPLEX, tl / 2.0, cv::Scalar(255, 255, 255),
                    tf, cv::LINE_AA);

        // Draw visible keypoints as filled circles. (KR: 가시 키포인트를 채워진 원으로 그린다.)
        for (int k = 0; k < static_cast<int>(d.kpts.size()); ++k) {
            const auto& kp = d.kpts[k];
            if (kp.score < KPT_DRAW_THRES) continue;
            cv::circle(img, cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)),
                       radius, POSE_PALETTE[KEYPOINT_PALETTE_IDX[k]], -1, cv::LINE_AA);
        }

        // Draw skeleton limbs between visible keypoint pairs. (KR: 가시 키포인트 쌍 사이에 스켈레톤 limb 를 그린다.)
        for (int i = 0; i < 19; ++i) {
            int a = POSE_SKELETON[i][0] - 1;  // skeleton is 1-indexed (KR: 스켈레톤은 1-기반)
            int b = POSE_SKELETON[i][1] - 1;
            if (a < 0 || b < 0 || a >= static_cast<int>(d.kpts.size()) ||
                b >= static_cast<int>(d.kpts.size())) {
                continue;
            }
            const auto& ka = d.kpts[a];
            const auto& kb = d.kpts[b];
            if (ka.score < KPT_DRAW_THRES || kb.score < KPT_DRAW_THRES) continue;
            cv::line(img, cv::Point(static_cast<int>(ka.x), static_cast<int>(ka.y)),
                     cv::Point(static_cast<int>(kb.x), static_cast<int>(kb.y)),
                     POSE_PALETTE[LIMB_PALETTE_IDX[i]], limb_thickness, cv::LINE_AA);
        }

        std::cout << "  person " << static_cast<int>(pct) << "% "
                  << "[" << x1 << "," << y1 << "," << x2 << "," << y2 << "]\n";
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.mxq> <image_path> <output_path>\n";
        return 1;
    }

    const std::string mxq_path = argv[1];
    const std::string image_path = argv[2];
    const std::string output_path = argv[3];

    ModelInfo cfg = make_yolo_pose_config();
    int nc = cfg.m_postprocess.num_classes;
    int nl = cfg.m_postprocess.num_layers;
    int reg_max = cfg.m_postprocess.reg_max;
    int num_keypoints = cfg.m_postprocess.num_keypoints;
    float conf_thres = cfg.m_postprocess.conf_thres;
    float iou_thres = cfg.m_postprocess.iou_thres;

    NPURunner model(mxq_path);
    auto shape = model.get_input_shape();
    std::cout << "Model input: " << shape[0] << "x" << shape[1] << "x"
              << shape[2] << "\n";

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return 1;
    }
    int img_h = img.rows;
    int img_w = img.cols;
    std::cout << "Image size: " << img_w << "x" << img_h << "\n";

    Transformer transformer;
    auto input = transformer.transform_uint8(img, cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto outputs = model.infer_uint8(std::move(input));
    auto t1 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Inference time: " << infer_ms << " ms\n";

    // DFL decode + keypoint decode + NMS, then rescale boxes and keypoints from letterbox space to original image coordinates
    // (KR: DFL 디코드 + 키포인트 디코드 + NMS 후 letterbox 좌표를 원본 이미지 좌표로 변환)
    YoloPoseDecoder decoder(nc, nl, IMG_SIZE, reg_max, num_keypoints, conf_thres,
                            iou_thres);
    auto dets = decoder.decode(outputs);
    YoloPoseDecoder::scale_to_original(dets, IMG_SIZE, img_h, img_w);
    std::cout << "Detections: " << dets.size() << "\n";

    draw_poses(img, dets);
    cv::imwrite(output_path, img);
    std::cout << "Result saved to: " << output_path << "\n";

    return 0;
}
