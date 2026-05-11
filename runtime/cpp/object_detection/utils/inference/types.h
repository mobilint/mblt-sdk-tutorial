// Shared type definitions for the inference pipeline.
// Declares PreProcessOps, Task, ImageSize, PreProcessInfo, PostProcessInfo, ModelInfo.
//
// (KR) 추론 파이프라인 공유 자료형 선언.
// PreProcessOps, Task, ImageSize, PreProcessInfo, PostProcessInfo, ModelInfo 를 선언한다.
#pragma once
#include <string>
#include <utility>
#include <variant>
#include <vector>

enum class PreProcessOps {
    RESIZE,      // fixed size or short-edge resize (KR: 고정 크기 또는 short-edge 리사이즈)
    CENTERCROP,  // center crop to target size (KR: 중앙 기준 잘라내기)
    NORMALIZE,   // pixel normalization: torch / tf / div255 (KR: 픽셀 정규화)
    YOLO,        // letterbox with pad value 114 (KR: pad=114 letterbox)
};

enum class Task {
    CLS,
    DET,
};

using ImageSize = std::variant<std::monostate, int, std::pair<int, int>>;

struct PreProcessInfo {
    PreProcessOps op;
    std::string style;
    ImageSize img_size{};
};

struct PostProcessInfo {
    Task task = Task::CLS;
    std::string type;
    int num_classes = 0;
    int num_layers = 0;
    int reg_max = 16;        // distribution focal loss bins (KR: DFL 분포 bin 수)
    float conf_thres = 0.f;
    float iou_thres = 0.f;
    std::vector<std::vector<std::vector<double>>> anchors;
};

struct ModelInfo {
    std::vector<PreProcessInfo> m_preprocess_list;
    PostProcessInfo m_postprocess;
};
