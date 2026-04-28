// =============================================================================
// parser.cc - YAML 파서 구현 (YAML parser implementation)
// =============================================================================
// OpenCV FileStorage 로 model_configs/<task>/*.yaml 을 파싱한다.
// =============================================================================
#include "parser.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>

static std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

static bool open_yaml_relaxed(const std::string& path, cv::FileStorage& fs) {
    try {
        fs.open(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
    } catch (const cv::Exception&) {
    }
    if (fs.isOpened()) return true;

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    std::string text((std::istreambuf_iterator<char>(ifs)), {});
    if (text.size() >= 3 && (unsigned char)text[0] == 0xEF &&
        (unsigned char)text[1] == 0xBB && (unsigned char)text[2] == 0xBF) {
        text.erase(0, 3);
    }
    if (text.rfind("%YAML:", 0) != 0) {
        text = std::string("%YAML:1.0\n") + text;
    }

    try {
        fs.open(text, cv::FileStorage::READ | cv::FileStorage::MEMORY |
                          cv::FileStorage::FORMAT_YAML);
    } catch (const cv::Exception&) {
    }
    return fs.isOpened();
}

static ImageSize read_img_size(const cv::FileNode& node) {
    cv::FileNode n = node["size"];
    if (n.empty()) n = node["img_size"];
    if (n.empty()) return std::monostate{};

    if (n.isSeq()) {
        std::vector<int> v;
        n >> v;
        if (v.size() >= 2) return std::pair<int, int>{v[0], v[1]};
        if (v.size() == 1) return v[0];
        return std::monostate{};
    }

    int s = 0;
    if (n.isInt()) {
        n >> s;
    } else if (n.isReal()) {
        double d = 0.0;
        n >> d;
        s = static_cast<int>(d);
    } else {
        n >> s;
    }
    return s;
}

static std::string read_string(const cv::FileNode& node) {
    if (node.empty()) return {};
    std::string s;
    node >> s;
    return s;
}

static Task parse_task(const std::string& s) {
    std::string t = lower_copy(s);
    if (t == "classification" || t == "image_classification") return Task::CLS;
    if (t == "object_detection" || t == "detection") return Task::DET;
    std::cerr << "[WARN] Unknown task: '" << s << "', defaulting to CLS\n";
    return Task::CLS;
}

static void parse_one_pre_block(const std::string& key, const cv::FileNode& node,
                                ModelInfo& config) {
    std::string lower = lower_copy(key);
    if (lower == "resize") {
        PreProcessInfo p{};
        p.op = PreProcessOps::RESIZE;
        p.img_size = read_img_size(node);
        p.style = read_string(node["interpolation"]);
        config.m_preprocess_list.push_back(std::move(p));
    } else if (lower == "centercrop" || lower == "center_crop") {
        PreProcessInfo p{};
        p.op = PreProcessOps::CENTERCROP;
        p.img_size = read_img_size(node);
        config.m_preprocess_list.push_back(std::move(p));
    } else if (lower == "normalize") {
        PreProcessInfo p{};
        p.op = PreProcessOps::NORMALIZE;
        p.style = read_string(node["style"]);
        config.m_preprocess_list.push_back(std::move(p));
    } else if (lower == "letterbox" || lower == "yolopre" || lower == "yolo") {
        PreProcessInfo p{};
        p.op = PreProcessOps::YOLO;
        p.img_size = read_img_size(node);
        config.m_preprocess_list.push_back(std::move(p));
    } else {
        std::cerr << "[WARN] Unknown pre op: '" << key << "' (skipped)\n";
    }
}

bool parse_yaml(const std::string& yaml_path, ModelInfo& config) {
    cv::FileStorage fs;
    if (!open_yaml_relaxed(yaml_path, fs)) {
        std::cerr << "Failed to open YAML: " << yaml_path << std::endl;
        return false;
    }

    config.m_preprocess_list.clear();
    cv::FileNode pre = fs["Pre-processing"];
    if (pre.empty()) {
        std::cerr << "Missing 'Pre-processing'\n";
        return false;
    }

    cv::FileNode pre_order = fs["Pre-order"];
    if (!pre_order.empty() && pre_order.isSeq()) {
        for (auto it = pre_order.begin(); it != pre_order.end(); ++it) {
            std::string name;
            (*it) >> name;
            if (name.empty()) continue;
            auto node = pre[name];
            if (node.empty()) {
                std::cerr << "[WARN] Pre-Order item '" << name << "' not found\n";
                continue;
            }
            parse_one_pre_block(name, node, config);
        }
    }

    cv::FileNode post = fs["Post-processing"];
    if (post.empty()) {
        std::cerr << "Missing 'Post-processing'\n";
        return false;
    }

    config.m_postprocess.task = parse_task(read_string(post["task"]));
    if (auto n = post["type"]; !n.empty()) n >> config.m_postprocess.type;
    if (auto n = post["nc"]; !n.empty()) n >> config.m_postprocess.num_classes;
    if (auto n = post["nl"]; !n.empty()) n >> config.m_postprocess.num_layers;
    if (auto n = post["reg_max"]; !n.empty()) n >> config.m_postprocess.reg_max;
    if (auto n = post["conf_thres"]; !n.empty()) n >> config.m_postprocess.conf_thres;
    if (auto n = post["iou_thres"]; !n.empty()) n >> config.m_postprocess.iou_thres;

    cv::FileNode an = post["anchors"];
    if (!an.empty() && an.type() == cv::FileNode::SEQ) {
        auto& anchors = config.m_postprocess.anchors;
        for (auto it = an.begin(); it != an.end(); ++it) {
            cv::FileNode layer_node = *it;
            std::vector<std::vector<double>> layer;
            if (layer_node.type() == cv::FileNode::SEQ) {
                std::vector<double> flat;
                layer_node >> flat;
                for (size_t i = 0; i + 1 < flat.size(); i += 2) {
                    layer.push_back({flat[i], flat[i + 1]});
                }
            }
            anchors.push_back(std::move(layer));
        }
        config.m_postprocess.num_layers = static_cast<int>(anchors.size());
    }
    return true;
}
