// Shared type definitions for the preprocess/postprocess pipeline.
// Declares PreProcessOps, Task, ImageSize, PreProcessInfo, PostProcessInfo, ModelInfo.
#pragma once
#include <string>
#include <utility>
#include <variant>
#include <vector>

enum class PreProcessOps {
  RESIZE,      // fixed-size or short-edge resize
  CENTERCROP,  // center crop to target size
  NORMALIZE,   // pixel normalization: torch / tf / div255
  YOLO,        // letterbox with pad value 114
};

enum class Task {
  CLS,
  DET,
  POSE,
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
  int reg_max = 16;       // distribution focal loss (DFL) bins
  int num_keypoints = 0;  // pose keypoints per detection (0 for non-pose)
  float conf_thres = 0.f;
  float iou_thres = 0.f;
  std::vector<std::vector<std::vector<double>>> anchors;
};

struct ModelInfo {
  std::vector<PreProcessInfo> m_preprocess_list;
  PostProcessInfo m_postprocess;
};
