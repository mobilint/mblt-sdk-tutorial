// =============================================================================
// parser.h - YAML 설정 파서 + ModelInfo 자료형 (YAML config parser + types)
// =============================================================================
// model_configs/<task>/*.yaml 을 ModelInfo 로 변환한다.
// types.h + parser.h 가 이 파일 하나로 통합되었다.
//
// Reads model_configs/<task>/*.yaml and produces ModelInfo.
// Combines former types.h + parser.h into a single header.
// =============================================================================
#pragma once
#include <string>
#include <utility>
#include <variant>
#include <vector>

enum class PreProcessOps {
    RESIZE,      // 고정 또는 short-edge 리사이즈
    CENTERCROP,  // 중앙 잘라내기
    NORMALIZE,   // 정규화 (torch / tf / div255)
    YOLO,        // YOLO letterbox (pad=114)
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
    int reg_max = 16;        // DFL reg_max (anchorless YOLO)
    float conf_thres = 0.f;
    float iou_thres = 0.f;
    std::vector<std::vector<std::vector<double>>> anchors;
};

struct ModelInfo {
    std::vector<PreProcessInfo> m_preprocess_list;
    PostProcessInfo m_postprocess;
};

// YAML -> ModelInfo. 성공 true, 실패 false.
bool parse_yaml(const std::string& yaml_path, ModelInfo& config);
