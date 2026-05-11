# 객체 검출 - C++ 추론 (ARIES + REGULUS)

단일 이미지에 대해 C++ NPU 추론을 실행하고 bounding box 를 시각화하는 예제입니다.
ultralytics v8/v9/v11 의 anchor-free Detect head 가 동일한 출력 layout
(3 stride x [reg_max*4 box + nc cls] = 144 채널) 을 갖기 때문에, 같은 binary 가
YOLO11m (ARIES) 와 YOLOv9m (REGULUS) MXQ 둘 다 처리합니다.

같은 `CMakeLists.txt` 로 **ARIES 네이티브 빌드** (x86_64 호스트 + NPU) 와
**REGULUS 크로스 컴파일** (x86_64 호스트 -> ARM64 타겟 보드) 두 경로를 모두 지원합니다.

## 파일 구조

- `infer_det.cc` - 추론 바이너리 소스 (NPU 추론, 후처리, bbox 시각화)
- `yolo_detect_config.h` - 앵커리스 YOLO detect 설정 (yolo11m / yolov9m P5, 80 클래스)
- `utils/` - 공용 추론 모듈 (NPURunner, Transformer, YoloDecoder)
- `CMakeLists.txt` - CMake 빌드 설정 (호스트 arch 자동 감지)

## 사전 준비

- [컴파일러 튜토리얼](../../../compilation/object_detection/README.KR.md) 결과의 MXQ 파일이 필요합니다:
  - **ARIES**: `yolo11m.mxq` (`model_compile.py` 산출물).
  - **REGULUS**: `yolov9m.mxq` (`model_compile_regulus.py` 산출물).

### 공통 요구 사항 (두 경로 모두)

- CMake >= 3.18
- C++17 컴파일러 (gcc / clang)
- `qbruntime` 라이브러리 (Mobilint NPU SDK 설치 시 함께 설치됨)

### ARIES 네이티브 빌드 (x86_64 호스트 + NPU)

호스트에 OpenCV 와 빌드 도구를 설치합니다 (Ubuntu / Debian 기준):

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

### REGULUS 크로스 컴파일 (x86_64 호스트 -> ARM64 타겟 보드)

벤더 크로스 컴파일 툴체인에는 OpenCV 와 `qbruntime` 이 포함되어 있습니다.
툴체인을 확인하고 환경을 활성화합니다:

```bash
ls /opt/crosstools/mobilint/                                   # 버전 디렉토리가 보여야 함
unset LD_LIBRARY_PATH                                          # 호스트 CUDA 등 충돌 방지
source /opt/crosstools/mobilint/{version}/{sdk}/environment-setup-cortexa53-mobilint-linux
echo $CXX                                                      # aarch64-mobilint-linux-g++ ...
```

툴체인이 설치되어 있지 않으면 [크로스 컴파일 준비](../README.KR.md) 를 참조해 먼저 설치합니다.

## 빌드

ARIES 네이티브와 REGULUS 크로스 컴파일 모두 같은 명령으로 빌드됩니다.
`CMakeLists.txt` 가 호스트 arch 를 감지해 `-march` 플래그를 자동 선택합니다.

```bash
cmake -B build -S .
cmake --build build -j
```

빌드가 끝나면 `build/infer-det` 바이너리가 생성됩니다.

바이너리의 아키텍처를 확인합니다:

```bash
file build/infer-det
# ARIES:   ELF 64-bit LSB executable, x86-64, ...
# REGULUS: ELF 64-bit LSB executable, ARM aarch64, ...
```

## 실행

샘플 이미지 `../rc/cr7.jpg` 가 저장소에 함께 들어 있습니다.

### ARIES (같은 호스트)

```bash
./build/infer-det ../../../compilation/object_detection/yolo11m.mxq ../rc/cr7.jpg result.jpg
```

### REGULUS (타겟 보드)

`build/infer-det`, `yolov9m.mxq`, `../rc/cr7.jpg` 를 타겟 보드로 복사한 뒤 실행합니다:

```bash
chmod +x infer-det
./infer-det yolov9m.mxq cr7.jpg result.jpg
```

## 출력 예시

```
Model input: 640x640x3
Image size: 980x652
Inference time: 21.5512 ms
Detections: 7
  person 94% [15,88,287,560]
  person 94% [436,61,717,567]
  sports ball 91% [238,521,318,598]
  person 83% [728,151,855,455]
  person 65% [576,253,645,420]
  person 34% [285,290,348,407]
  person 27% [921,313,977,440]
Result saved to: result.jpg
```

결과 이미지 `result.jpg` 에는 원본 위에 bounding box 와 클래스 라벨이 그려져 저장됩니다.
