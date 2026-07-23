// End-to-end object detection inference on Mobilint NPU with bounding-box visualization.
// Preprocessing (letterbox + BGR->RGB) is handled by Preprocessor.
// --input   : uint8 feeds the fused-normalization MXQ; float applies /255 here for the !uint8 MXQ.
// --inf-func : chw -> CHW buffer + Model::inferCHW (default) | hwc -> HWC buffer + Model::infer.
//              Diagnostic switch; the selected path runs as-is with no fallback so its
//              status/result is visible. These YOLO MXQ only decode correctly with chw.
// Pipeline: load MXQ -> transform -> NPU infer(CHW) -> DFL decode -> NMS -> draw boxes.
//
// Usage:
//   ./infer-det <model.mxq> <image_path> <output_path> [--input uint8|float] [--inf-func chw|hwc]
//
// Examples:
//   ./infer-det yolo11m.mxq cr7.jpg result.jpg                 # ARIES / REGULUS regulus-rb
//   ./infer-det yolo11m.mxq cr7.jpg result.jpg --inf-func hwc  # try the HWC+infer path
//   ./infer-det yolov9m.mxq cr7.jpg result.jpg                 # REGULUS regulus-ra (older)

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <qbruntime/qbruntime.h>

#include "decode.h"
#include "preprocessor.h"
#include "yolo_detect_config.h"

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
    // Positional: <model.mxq> <image_path> <output_path>.
    // Optional: --input uint8|float  (input element type)
    //           --inf-func chw|hwc   (chw -> Model::inferCHW, hwc -> Model::infer)
    std::vector<std::string> pos;
    std::string input_type = "uint8";
    std::string inf_func = "chw";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--input" && i + 1 < argc) {
            input_type = argv[++i];
        } else if (a == "--inf-func" && i + 1 < argc) {
            inf_func = argv[++i];
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() != 3 || (input_type != "uint8" && input_type != "float") ||
        (inf_func != "chw" && inf_func != "hwc")) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.mxq> <image_path> <output_path>"
                     " [--input uint8|float] [--inf-func chw|hwc]\n";
        return 1;
    }
    const std::string mxq_path = pos[0];
    const std::string image_path = pos[1];
    const std::string output_path = pos[2];
    std::cout << "Input mode: " << input_type << "\n";
    std::cout << "Inference func: "
              << (inf_func == "chw" ? "inferCHW (CHW buffer)" : "infer (HWC buffer)") << "\n";

    ModelInfo cfg = make_yolo_detect_config();

    // Load the MXQ onto the NPU. Single-core mode (Cluster0/Core0) handles both
    // ARIES multi-mode and REGULUS single-mode MXQ files.
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
    int img_h = img.rows, img_w = img.cols;
    std::cout << "Image size: " << img_w << "x" << img_h << "\n";

    // Build the input buffer and run the selected path (no fallback).
    //   uint8 : raw letterboxed pixels (normalization fused in the MXQ).
    //   float : letterboxed pixels /255 (the !uint8 MXQ has no fused normalization).
    //   chw   : CHW buffer -> Model::inferCHW.   hwc : HWC buffer -> Model::infer.
    Preprocessor preprocessor;
    std::vector<std::vector<float>> outputs;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (inf_func == "chw") {
        if (input_type == "float") {
            auto input = preprocessor.transform_float_chw(img, cfg);
            outputs = model->inferCHW({input.get()}, sc);
        } else {
            auto input = preprocessor.transform_uint8(img, cfg);
            outputs = model->inferCHW({input.get()}, sc);
        }
    } else {  // hwc
        if (input_type == "float") {
            auto input = preprocessor.transform_float_hwc(img, cfg);
            outputs = model->infer({input.get()}, sc);
        } else {
            auto input = preprocessor.transform_uint8_hwc(img, cfg);
            outputs = model->infer({input.get()}, sc);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    // Report the raw inference status; do not fall back to another path.
    // Prints the StatusCode enum value (0 = OK); cross-reference qbruntime/status_code.h.
    std::cout << "Inference status: " << (!sc ? "ERROR" : "OK") << " (code "
              << static_cast<int>(sc) << ")\n";
    std::cout << "Inference time: "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

    // DFL decode + NMS, then rescale boxes from letterbox space to original image coordinates.
    YoloDecoder decoder(cfg.m_postprocess.num_classes, cfg.m_postprocess.num_layers,
                        IMG_SIZE, cfg.m_postprocess.reg_max,
                        cfg.m_postprocess.conf_thres, cfg.m_postprocess.iou_thres);
    auto dets = decoder.decode(outputs);
    YoloDecoder::scale_to_original(dets, IMG_SIZE, img_h, img_w);
    std::cout << "Detections: " << dets.size() << "\n";

    draw_detections(img, dets);
    cv::imwrite(output_path, img);
    std::cout << "Result saved to: " << output_path << "\n";

    model->dispose();
    return 0;
}
