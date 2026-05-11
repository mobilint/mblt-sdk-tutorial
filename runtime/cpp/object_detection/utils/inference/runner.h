// NPU runner wrapping qbruntime Accelerator and Model, plus a NumPy .npy save utility.
// NPU access is only available on the target ARM64 board; host builds will link-fail on qbruntime symbols.
//
// (KR) qbruntime Accelerator 와 Model 을 래핑하는 NPU 실행기, NumPy .npy 저장 유틸 포함.
// NPU 접근은 대상 ARM64 보드에서만 가능하며, 호스트 빌드는 qbruntime 심벌에서 링크 오류가 발생한다.
#pragma once
#include <qbruntime/qbruntime.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class NPURunner {
public:
    explicit NPURunner(const std::string& model_path);
    ~NPURunner();

    // Runs inference with a float HWC input buffer; returns output tensors as flat float vectors.
    // (KR: float HWC 입력 버퍼로 추론 실행; 출력 텐서를 평면 float 벡터 목록으로 반환.)
    std::vector<std::vector<float>> infer(std::unique_ptr<float[]> input);

    // Runs inference with a uint8 CHW input buffer compiled with Uint8InputConfig.
    // (KR: Uint8InputConfig 로 컴파일된 MXQ 에 대해 uint8 CHW 입력 버퍼로 추론 실행.)
    std::vector<std::vector<float>> infer_uint8(std::unique_ptr<uint8_t[]> input);

    // Returns model input shape as {height, width, channel}.
    // (KR: 모델 입력 shape 를 {height, width, channel} 순서로 반환.)
    std::vector<int> get_input_shape() const;

private:
    mobilint::StatusCode sc_;
    std::unique_ptr<mobilint::Accelerator> acc_;
    std::unique_ptr<mobilint::Model> model_;
};

// Saves a single float32 tensor to a .npy file (NumPy v1.0 format, no zip, no external libraries).
// Format: magic(\x93NUMPY) + version(1.0) + ASCII dict header + raw float32 data row-major.
// shape entries follow NumPy dimension order (e.g., {1, 64, 80, 80}).
// (KR: 단일 float32 텐서를 .npy 파일로 저장(NumPy v1.0, zip 없음, 외부 라이브러리 없음).
// shape 는 NumPy 차원 순서를 따른다(예: {1, 64, 80, 80}).)
void save_npy(const std::string& path, const float* data,
              const std::vector<size_t>& shape);
