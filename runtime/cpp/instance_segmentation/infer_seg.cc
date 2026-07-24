// End-to-end instance segmentation inference on Mobilint NPU with mask + bounding-box visualization.
// Preprocessing (letterbox + BGR->RGB + HWC->CHW) is handled by Preprocessor.
// Input mode via --input-dtype: uint8 feeds the fused-normalization MXQ; float applies /255 here for the !uint8 MXQ.
// Pipeline: load MXQ -> transform (HWC) -> NPU infer -> DFL decode + NMS -> mask assembly -> draw masks + boxes.
//
// Usage:
//   ./infer-seg <model.mxq> <image_path> <output_path> [--input-dtype uint8|float]
//
// Examples:
//   ./infer-seg yolo11m-seg.mxq cr7.jpg result.jpg   # ARIES / REGULUS regulus-rb
//   ./infer-seg yolov8m-seg.mxq cr7.jpg result.jpg   # REGULUS regulus-ra (older)

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <qbruntime/qbruntime.h>

#include "decode.h"
#include "preprocessor.h"
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
    // uint8 : normalization fused into the MXQ (uint8-input model).
    // float : preprocessing NOT fused (!uint8 model); this program normalizes (/255) and feeds float.
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

    ModelInfo cfg = make_yolo_seg_config();
    int nc = cfg.m_postprocess.num_classes;
    int nl = cfg.m_postprocess.num_layers;
    int reg_max = cfg.m_postprocess.reg_max;
    int num_mask_coeffs = cfg.m_postprocess.num_mask_coeffs;
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

    // Pick the layout automatically from the MXQ's declared input shape (no flag):
    //   channel-last (HWC)  -> Model::infer.  All tutorial MXQ are here.
    //   channel-first (CHW) -> Model::inferCHW, then transpose outputs to HWC for the decoder.
    //   uint8 : normalization fused in the MXQ.  float : /255 applied here for the !uint8 MXQ.
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
    double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Inference time: " << infer_ms << " ms\n";

    // DFL decode + NMS + prototype extraction, then assemble masks before rescaling boxes.
    // Masks need boxes in letterbox space for cropping, so assemble first, then scale boxes for drawing.
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

    model->dispose();
    return 0;
}
