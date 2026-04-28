// Object detection inference on Mobilint NPU with post-processing and visualization.
//
// Uses inference-regulus modules (NPURunner, Transformer, YoloDecoder).
// Preprocessing: letterbox + BGR2RGB + HWC->CHW (via Transformer)
// Normalization: fused into MXQ model (Uint8InputConfig)
// Input: uint8 letterboxed CHW image
//
// Usage:
//   ./infer-det <model.mxq> <image_path> <output_path>
//
// Example:
//   ./infer-det yolov9m.mxq example.jpg result.jpg

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "decode.h"
#include "runner.h"
#include "transform.h"
#include "yolov9m_config.h"

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

void draw_detections(cv::Mat& img,
                     const std::vector<YoloDecoder::Detection>& dets) {
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

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                      cv::Scalar(0, 255, 0), 2);

        std::string label = name + " " + std::to_string(pct) + "%";
        cv::putText(img, label, cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

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

    // Model config (hardcoded, no YAML)
    ModelInfo cfg = make_yolov9m_config();
    int nc = cfg.m_postprocess.num_classes;
    int nl = cfg.m_postprocess.num_layers;
    int reg_max = cfg.m_postprocess.reg_max;
    float conf_thres = cfg.m_postprocess.conf_thres;
    float iou_thres = cfg.m_postprocess.iou_thres;

    // 1) Load MXQ model onto NPU
    NPURunner model(mxq_path);
    auto shape = model.get_input_shape();
    std::cout << "Model input: " << shape[0] << "x" << shape[1] << "x"
              << shape[2] << "\n";

    // 2) Load image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << "\n";
        return 1;
    }
    int img_h = img.rows;
    int img_w = img.cols;
    std::cout << "Image size: " << img_w << "x" << img_h << "\n";

    // 3) Preprocess: letterbox + BGR2RGB + HWC->CHW (uint8)
    Transformer transformer;
    auto input = transformer.transform_uint8(img, cfg);

    // 4) Run NPU inference
    auto t0 = std::chrono::high_resolution_clock::now();
    auto outputs = model.infer_uint8(std::move(input));
    auto t1 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Inference time: " << infer_ms << " ms\n";

    // 5) Post-process: DFL decode + NMS + rescale to original image
    YoloDecoder decoder(nc, nl, IMG_SIZE, reg_max, conf_thres, iou_thres);
    auto dets = decoder.decode(outputs);
    YoloDecoder::scale_to_original(dets, IMG_SIZE, img_h, img_w);
    std::cout << "Detections: " << dets.size() << "\n";

    // 6) Draw bounding boxes and save result
    draw_detections(img, dets);
    cv::imwrite(output_path, img);
    std::cout << "Result saved to: " << output_path << "\n";

    return 0;
}
