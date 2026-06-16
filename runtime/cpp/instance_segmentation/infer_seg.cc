// End-to-end instance segmentation inference on Mobilint NPU with mask + bounding-box visualization.
// Preprocessing (letterbox + BGR->RGB + HWC->CHW) is handled by Transformer.
// Normalization is fused into the MXQ model (uint8 input), so no float scaling is needed here.
// Pipeline: load MXQ -> transform uint8 -> NPU infer -> DFL decode + NMS -> mask assembly -> draw masks + boxes.
//
// Usage:
//   ./infer-seg <model.mxq> <image_path> <output_path>
//
// Examples:
//   ./infer-seg yolo11m-seg.mxq cr7.jpg result.jpg   # ARIES
//   ./infer-seg yolov8m-seg.mxq cr7.jpg result.jpg   # REGULUS
//
// (KR) Mobilint NPU 에서 인스턴스 분할 추론을 실행하고 마스크와 바운딩 박스를 이미지에 그린다.
// 전처리(letterbox + BGR->RGB + HWC->CHW)는 Transformer 가 담당한다.
// 정규화는 MXQ 모델에 퓨즈되어 있어(uint8 입력) 별도 float 변환이 필요 없다.
// 파이프라인: MXQ 로드 -> uint8 변환 -> NPU 추론 -> DFL 디코드 + NMS -> 마스크 조립 -> 마스크 + 박스 시각화.

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "decode.h"
#include "runner.h"
#include "transform.h"
#include "yolo_seg_config.h"

static const std::vector<std::string> COCO_LABELS = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush",
};

// COCO detection palette as BGR triples, mirroring coco.py DET_PALETTE (which is in RGB).
// Index by class id; entries are stored BGR so cv:: draws the same color the Python tutorial uses.
// (KR) COCO 탐지 팔레트 BGR 삼중값, coco.py 의 DET_PALETTE(RGB) 를 그대로 옮김.
// 클래스 id 로 색인하며 cv:: 가 Python 튜토리얼과 같은 색을 그리도록 BGR 로 저장한다.
static const std::vector<cv::Scalar> COCO_PALETTE = {
    {60, 20, 220},   {32, 11, 119},   {142, 0, 0},     {230, 0, 0},
    {228, 0, 106},   {100, 60, 0},    {100, 80, 0},    {70, 0, 0},
    {192, 0, 0},     {30, 170, 250},  {30, 170, 100},  {0, 220, 220},
    {175, 116, 175}, {30, 0, 250},    {42, 42, 165},   {255, 77, 255},
    {252, 226, 0},   {255, 182, 182}, {0, 82, 0},      {157, 166, 120},
    {0, 76, 110},    {255, 57, 174},  {0, 100, 199},   {118, 0, 72},
    {240, 179, 255}, {92, 125, 0},    {151, 0, 209},   {182, 208, 188},
    {176, 220, 0},   {164, 99, 255},  {73, 0, 92},     {255, 129, 133},
    {255, 180, 78},  {0, 228, 0},     {243, 255, 174}, {255, 89, 45},
    {103, 134, 134}, {174, 148, 145}, {186, 208, 255}, {255, 226, 197},
    {1, 134, 171},   {54, 63, 109},   {255, 138, 207}, {95, 0, 151},
    {61, 80, 9},     {51, 105, 84},   {105, 65, 74},   {102, 196, 166},
    {210, 195, 208}, {65, 109, 255},  {149, 143, 0},   {194, 0, 179},
    {106, 99, 209},  {0, 121, 5},     {205, 255, 227}, {208, 186, 147},
    {1, 69, 153},    {161, 95, 3},    {0, 255, 163},   {170, 0, 119},
    {199, 182, 0},   {120, 165, 0},   {88, 130, 183},  {0, 32, 95},
    {135, 114, 130}, {133, 129, 110}, {118, 74, 166},  {185, 142, 219},
    {114, 210, 79},  {62, 90, 178},   {15, 70, 65},    {115, 167, 127},
    {106, 105, 59},  {45, 108, 142},  {0, 172, 196},   {80, 54, 95},
    {255, 76, 128},  {1, 57, 201},    {122, 0, 246},   {208, 162, 191},
};

static cv::Scalar class_color(int cls) {
    int n = static_cast<int>(COCO_PALETTE.size());
    return COCO_PALETTE[(cls >= 0 && cls < n) ? cls : 0];
}

// Alpha-blends one colored mask per detection onto the image, mirroring visualize.py draw_masks (alpha=0.3).
// (KR) 탐지별 색상 마스크를 alpha 블렌딩으로 이미지에 합성, visualize.py 의 draw_masks (alpha=0.3) 를 따른다.
static void draw_masks(cv::Mat& img,
                       const std::vector<cv::Mat>& masks,
                       const std::vector<YoloSegDecoder::Detection>& dets,
                       float alpha = 0.3f) {
    for (size_t i = 0; i < masks.size(); ++i) {
        const cv::Mat& m = masks[i];
        cv::Scalar color = class_color(dets[i].cls);
        for (int y = 0; y < img.rows; ++y) {
            const uint8_t* mrow = m.ptr<uint8_t>(y);
            cv::Vec3b* irow = img.ptr<cv::Vec3b>(y);
            for (int x = 0; x < img.cols; ++x) {
                if (mrow[x] == 0) continue;
                for (int c = 0; c < 3; ++c) {
                    float blended = irow[x][c] * (1.0f - alpha) +
                                    static_cast<float>(color[c]) * alpha;
                    irow[x][c] = static_cast<uint8_t>(std::min(255.0f, blended));
                }
            }
        }
    }
}

// Draws bounding boxes and class labels, mirroring visualize.py draw_boxes.
// (KR) 바운딩 박스와 클래스 라벨을 그린다, visualize.py 의 draw_boxes 를 따른다.
static void draw_boxes(cv::Mat& img,
                       const std::vector<YoloSegDecoder::Detection>& dets) {
    for (const auto& d : dets) {
        int x1 = static_cast<int>(d.x1);
        int y1 = static_cast<int>(d.y1);
        int x2 = static_cast<int>(d.x2);
        int y2 = static_cast<int>(d.y2);
        int pct = static_cast<int>(d.conf * 100);
        const std::string& name =
            (d.cls >= 0 && d.cls < static_cast<int>(COCO_LABELS.size()))
                ? COCO_LABELS[d.cls]
                : COCO_LABELS[0];
        cv::Scalar color = class_color(d.cls);

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        std::string label = name + " " + std::to_string(pct) + "%";
        cv::putText(img, label, cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        std::cout << "  " << name << " " << pct << "% "
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

    ModelInfo cfg = make_yolo_seg_config();
    int nc = cfg.m_postprocess.num_classes;
    int nl = cfg.m_postprocess.num_layers;
    int reg_max = cfg.m_postprocess.reg_max;
    int num_mask_coeffs = cfg.m_postprocess.num_mask_coeffs;
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

    // DFL decode + NMS + prototype extraction, then assemble masks before rescaling boxes.
    // Masks need boxes in letterbox space for cropping, so assemble first, then scale boxes for drawing.
    // (KR: DFL 디코드 + NMS + prototype 추출 후 박스 rescale 전에 마스크를 조립한다.
    // 마스크 crop 은 letterbox 좌표 박스가 필요하므로 마스크를 먼저 조립하고, 이후 박스를 그리기 좌표로 변환한다.)
    YoloSegDecoder decoder(nc, nl, IMG_SIZE, reg_max, num_mask_coeffs,
                           conf_thres, iou_thres);
    std::vector<float> proto;
    int proto_c = 0, proto_h = 0, proto_w = 0;
    auto dets = decoder.decode(outputs, proto, proto_c, proto_h, proto_w);
    std::cout << "Detections: " << dets.size() << "\n";

    auto masks = decoder.assemble_masks(dets, proto, proto_c, proto_h, proto_w,
                                        img_h, img_w);
    YoloSegDecoder::scale_to_original(dets, IMG_SIZE, img_h, img_w);

    draw_masks(img, masks, dets);
    draw_boxes(img, dets);
    cv::imwrite(output_path, img);
    std::cout << "Result saved to: " << output_path << "\n";

    return 0;
}
