// NPU runner wrapping qbruntime Accelerator and Model, plus a NumPy .npy save utility.
// Supported execution paths:
//   - ARIES (x86_64 host with the Mobilint NPU): native build links against the
//     host's libqbruntime and runs inference on the host NPU.
//   - REGULUS (ARM64 target board): the binary is cross-compiled on an x86_64
//     host using the Mobilint SDK toolchain, then deployed to the board where
//     libqbruntime is preinstalled.
// Both paths share this header and runner.cc; only -march and the toolchain
// differ.
//
// (KR) qbruntime Accelerator 와 Model 을 래핑하는 NPU 실행기, NumPy .npy 저장 유틸 포함.
// 지원 실행 경로:
//   - ARIES (x86_64 호스트 + Mobilint NPU): 호스트에서 네이티브 빌드해 호스트
//     의 libqbruntime 과 링크, 호스트 NPU 로 추론한다.
//   - REGULUS (ARM64 타겟 보드): x86_64 호스트에서 Mobilint SDK 툴체인으로
//     크로스 컴파일한 뒤, libqbruntime 이 사전 설치된 보드에 배포해 실행한다.
// 두 경로는 이 헤더와 runner.cc 를 공유하며 -march 와 툴체인만 다르다.
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
