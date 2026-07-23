// End-to-end pose estimation inference on Mobilint NPU with skeleton visualization.
// Preprocessing (letterbox + BGR->RGB + HWC->CHW) is handled by Preprocessor.
// Input mode via --input: uint8 feeds the fused-normalization MXQ; float applies /255 here for the !uint8 MXQ.
// Pipeline: load MXQ -> transform (CHW) -> NPU inferCHW -> DFL decode -> keypoint decode -> NMS -> draw boxes + skeleton.
//
// Usage:
//   ./infer-pose <model.mxq> <image_path> <output_path> [--input uint8|float]
//
// Examples:
//   ./infer-pose yolo11m-pose.mxq cr7.jpg result.jpg   # ARIES / REGULUS regulus-rb
//   ./infer-pose yolov8m-pose.mxq cr7.jpg result.jpg   # REGULUS regulus-ra (older)

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <qbruntime/qbruntime.h>

#include "decode.h"
#include "preprocessor.h"
#include "yolo_pose_config.h"

// COCO 17-keypoint skeleton as 1-indexed keypoint pairs (mirrors coco.py POSE_SKELETON).
static const int POSE_SKELETON[19][2] = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
    {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
    {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7},
};

// Base 20-color pose palette in RGB (mirrors coco.py POSE_PALETTE).
static const cv::Scalar POSE_PALETTE[20] = {
    {255, 128, 0},   {255, 153, 51},  {255, 178, 102}, {230, 230, 0},
    {255, 153, 255}, {153, 204, 255}, {255, 102, 255}, {255, 51, 255},
    {102, 178, 255}, {51, 153, 255},  {255, 153, 153}, {255, 102, 102},
    {255, 51, 51},   {153, 255, 153}, {102, 255, 102}, {51, 255, 51},
    {0, 255, 0},     {0, 0, 255},     {255, 0, 0},     {255, 255, 255},
};

// Palette index per skeleton limb (mirrors coco.py LIMB_PALLETE).
static const int LIMB_PALETTE_IDX[19] = {
    9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16};

// Palette index per keypoint (mirrors coco.py KEYPOINT_PALLETE).
static const int KEYPOINT_PALETTE_IDX[17] = {
    16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9};

// Keypoint visibility threshold for drawing (mirrors draw_kpts default conf=0.5 in visualize.py).
static const float KPT_DRAW_THRES = 0.5f;

void draw_poses(cv::Mat& img,
                const std::vector<YoloPoseDecoder::Detection>& dets) {
    int h = img.rows;
    int w = img.cols;
    // Line/font thickness scales with image size (mirrors visualize.py draw_boxes).
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

        // Draw visible keypoints as filled circles.
        for (int k = 0; k < static_cast<int>(d.kpts.size()); ++k) {
            const auto& kp = d.kpts[k];
            if (kp.score < KPT_DRAW_THRES) continue;
            cv::circle(img, cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)),
                       radius, POSE_PALETTE[KEYPOINT_PALETTE_IDX[k]], -1, cv::LINE_AA);
        }

        // Draw skeleton limbs between visible keypoint pairs.
        for (int i = 0; i < 19; ++i) {
            int a = POSE_SKELETON[i][0] - 1;  // skeleton is 1-indexed
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
    // Positional: <model.mxq> <image_path> <output_path>. Optional: --input uint8|float
    // uint8 : normalization fused into the MXQ (uint8-input model).
    // float : preprocessing NOT fused (!uint8 model); this program normalizes (/255) and feeds float.
    std::vector<std::string> pos;
    std::string input_type = "uint8";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--input" && i + 1 < argc) {
            input_type = argv[++i];
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() != 3 || (input_type != "uint8" && input_type != "float")) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.mxq> <image_path> <output_path> [--input uint8|float]\n";
        return 1;
    }

    const std::string mxq_path = pos[0];
    const std::string image_path = pos[1];
    const std::string output_path = pos[2];
    std::cout << "Input mode: " << input_type << "\n";

    ModelInfo cfg = make_yolo_pose_config();
    int nc = cfg.m_postprocess.num_classes;
    int nl = cfg.m_postprocess.num_layers;
    int reg_max = cfg.m_postprocess.reg_max;
    int num_keypoints = cfg.m_postprocess.num_keypoints;
    float conf_thres = cfg.m_postprocess.conf_thres;
    float iou_thres = cfg.m_postprocess.iou_thres;

    mobilint::StatusCode sc;
    auto acc = mobilint::Accelerator::create(sc);
    if (!sc) { std::cerr << "Failed to create accelerator\n"; return 1; }
    mobilint::ModelConfig mc;
    mc.setSingleCoreMode({mobilint::CoreId{mobilint::Cluster::Cluster0,
                                           mobilint::Core::Core0}});
    auto model = mobilint::Model::create(mxq_path, mc, sc);
    if (!sc) { std::cerr << "Failed to load model: " << mxq_path << "\n"; return 1; }
    sc = model->launch(*acc);
    if (!sc) { std::cerr << "Failed to launch model\n"; return 1; }

    // Model input shape as declared in the MXQ (HWC order, matches mxqtool "Shape").
    const auto& in_shapes = model->getModelInputShape();
    for (size_t i = 0; i < in_shapes.size(); ++i) {
        std::cout << "Model input shape[" << i << "]: [";
        for (size_t j = 0; j < in_shapes[i].size(); ++j)
            std::cout << in_shapes[i][j] << (j + 1 < in_shapes[i].size() ? ", " : "");
        std::cout << "]\n";
    }

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return 1;
    }
    int img_h = img.rows;
    int img_w = img.cols;
    std::cout << "Image size: " << img_w << "x" << img_h << "\n";

    Preprocessor preprocessor;
    std::vector<std::vector<float>> outputs;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (input_type == "float") {
        // !uint8 MXQ: /255 normalization is not fused. transform_float_chw emits a CHW /255 float
        // buffer (same layout as the working uint8 path), fed via inferCHW.
        auto input = preprocessor.transform_float_chw(img, cfg);
        outputs = model->inferCHW({input.get()}, sc);
    } else {
        auto input = preprocessor.transform_uint8(img, cfg);
        outputs = model->inferCHW({input.get()}, sc);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Inference time: " << infer_ms << " ms\n";

    // DFL decode + keypoint decode + NMS, then rescale boxes/keypoints from letterbox space to original image coordinates.
    YoloPoseDecoder decoder(nc, nl, IMG_SIZE, reg_max, num_keypoints, conf_thres,
                            iou_thres);
    auto dets = decoder.decode(outputs);
    YoloPoseDecoder::scale_to_original(dets, IMG_SIZE, img_h, img_w);
    std::cout << "Detections: " << dets.size() << "\n";

    draw_poses(img, dets);
    cv::imwrite(output_path, img);
    std::cout << "Result saved to: " << output_path << "\n";

    model->dispose();
    return 0;
}
