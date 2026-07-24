// End-to-end object detection inference on Mobilint NPU with bounding-box visualization.
// Preprocessing (letterbox + BGR->RGB) is handled by Preprocessor.
// --input-dtype : uint8 feeds the fused-normalization MXQ; float applies /255 here for the !uint8 MXQ.
// Layout is chosen automatically from getModelInputShape(): channel-last (HWC) MXQ use Model::infer;
// channel-first (CHW) MXQ use Model::inferCHW and the outputs are transposed back to HWC, so the
// decoder always reads HWC. The tutorial MXQ are all channel-last.
// Pipeline: load MXQ -> transform -> NPU infer/inferCHW -> (to HWC) -> DFL decode -> NMS -> draw boxes.
//
// Usage:
//   ./infer-det <model.mxq> <image_path> <output_path> [--input-dtype uint8|float]
//
// Examples:
//   ./infer-det yolo11m.mxq cr7.jpg result.jpg   # ARIES / REGULUS regulus-rb
//   ./infer-det yolov9m.mxq cr7.jpg result.jpg   # REGULUS regulus-ra

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

// Transpose a channel-first output tensor [C,H,W] to channel-last [H,W,C] so the HWC decoder
// can read it uniformly. Used only for channel-first MXQ (see the layout note at the top).
static mobilint::NDArray<float> chw_to_hwc(const mobilint::NDArray<float>& src,
                                           mobilint::StatusCode& sc) {
    const auto& s = src.shape();
    if (s.size() != 3) return src;  // only 3D feature maps need reordering
    const int64_t C = s[0], H = s[1], W = s[2];
    mobilint::NDArray<float> dst({H, W, C}, sc);
    const float* p = src.data();
    float* q = dst.data();
    for (int64_t c = 0; c < C; ++c)
        for (int64_t hw = 0; hw < H * W; ++hw)
            q[hw * C + c] = p[c * H * W + hw];  // [C,H*W] -> [H*W,C]
    return dst;
}

int main(int argc, char** argv) {
    // Positional: <model.mxq> <image_path> <output_path>. Optional: --input-dtype uint8|float
    std::vector<std::string> pos;
    std::string input_type = "uint8";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--input-dtype" && i + 1 < argc) {
            input_type = argv[++i];
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() != 3 || (input_type != "uint8" && input_type != "float")) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.mxq> <image_path> <output_path> [--input-dtype uint8|float]\n";
        return 1;
    }
    const std::string mxq_path = pos[0];
    const std::string image_path = pos[1];
    const std::string output_path = pos[2];
    std::cout << "Input mode: " << input_type << "\n";

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

    // Pick the layout automatically from the MXQ's declared input shape (no flag):
    //   channel-last (HWC)  -> Model::infer, NPU-native, no transpose.  All tutorial MXQ are here.
    //   channel-first (CHW) -> Model::inferCHW, then transpose outputs to HWC for the decoder.
    // Buffers/outputs use non-owning NDArray views/results so the qbruntime skips input/output
    // copies and output zero-fill.
    //   uint8 : raw letterboxed pixels (normalization fused in the MXQ).
    //   float : letterboxed pixels /255 (the !uint8 MXQ has no fused normalization).
    Preprocessor preprocessor;
    const std::vector<int64_t> in_shape(in_shapes[0].begin(), in_shapes[0].end());
    const bool channel_last = !in_shape.empty() && in_shape.back() == 3;
    std::vector<mobilint::NDArray<float>> outputs;
    auto t0 = std::chrono::high_resolution_clock::now();
    if (channel_last) {
        if (input_type == "float") {
            auto input = preprocessor.transform_float(img, cfg);
            std::vector<mobilint::NDArray<float>> in{mobilint::NDArray<float>(input.get(), in_shape)};
            outputs = model->infer(in, sc);
        } else {
            auto input = preprocessor.transform_uint8(img, cfg);
            std::vector<mobilint::NDArray<uint8_t>> in{mobilint::NDArray<uint8_t>(input.get(), in_shape)};
            outputs = model->infer(in, sc);
        }
    } else {
        if (input_type == "float") {
            auto input = preprocessor.transform_float_chw(img, cfg);
            std::vector<mobilint::NDArray<float>> in{mobilint::NDArray<float>(input.get(), in_shape)};
            outputs = model->inferCHW(in, sc);
        } else {
            auto input = preprocessor.transform_uint8_chw(img, cfg);
            std::vector<mobilint::NDArray<uint8_t>> in{mobilint::NDArray<uint8_t>(input.get(), in_shape)};
            outputs = model->inferCHW(in, sc);
        }
        for (auto& o : outputs) o = chw_to_hwc(o, sc);  // normalize CHW outputs to HWC
    }
    if (!sc) { std::cerr << "Inference failed (status " << static_cast<int>(sc) << ")\n"; return 1; }
    auto t1 = std::chrono::high_resolution_clock::now();
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
