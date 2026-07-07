# C++ 이미지 분류 런타임

이 튜토리얼은 C++ `qbruntime` API로 컴파일된 ResNet-50 MXQ 모델을 실행하는 방법을 설명합니다.

시작하기 전에 [../../../compilation/image_classification/README.KR.md](../../../compilation/image_classification/README.KR.md)의 컴파일 과정을 먼저 완료하세요. 이 런타임 예제는 컴파일된 `resnet50.mxq` 모델을 사용합니다.

## 사전 준비

다음 구성 요소가 준비되어 있어야 합니다.

- Mobilint `qbruntime`
- OpenCV 개발 라이브러리
- C++17 컴파일러
- CMake `3.21` 이상
- 해당 컴파일 튜토리얼에서 생성한 `resnet50.mxq`
- 이 디렉토리에 포함된 `imagenet_labels.txt`

Ubuntu 또는 Debian 기반 ARIES 네이티브 빌드에서는 다음과 같이 설치할 수 있습니다.

```bash
apt-get update
apt-get install -y build-essential cmake libopencv-dev
```

REGULUS 크로스 컴파일에서는 먼저 [../README.KR.md](../README.KR.md)에 정리된 Mobilint 툴체인을 활성화하세요.

## 개요

`infer_cls.cc`의 런타임 흐름은 다음 단계를 따릅니다.

1. `imagenet_labels.txt`에서 ImageNet 라벨을 읽습니다.
2. `qbruntime`로 컴파일된 MXQ 모델을 로드합니다.
3. 입력 이미지를 읽고 ResNet 전처리를 적용합니다.
4. Mobilint NPU에서 추론을 실행합니다.
5. 상위 5개 클래스 예측 결과를 출력합니다.

모델 입력은 `uint8`이며, 정규화는 컴파일된 MXQ 모델에 이미 융합되어 있다고 가정합니다.

## 이 튜토리얼의 파일

- `infer_cls.cc`: 전체 이미지 분류 파이프라인을 실행하고 상위 5개 예측 결과를 출력합니다.
- `imagenet_labels.txt`: ImageNet 1000개 클래스 라벨 파일입니다.
- `CMakeLists.txt`: `infer-cls` 실행 파일을 빌드합니다.

## 프로그램 동작 방식

프로그램은 다음 명령줄 형식을 사용합니다.

```bash
./infer-cls <model.mxq> <image_path> <labels_file>
```

입력 이미지는 다음 방식으로 전처리됩니다.

- 짧은 변을 `256`으로 리사이즈
- `224x224` center crop 적용
- BGR에서 RGB로 변환

추론이 끝나면 출력 로그잇을 정렬하고 상위 5개 클래스 ID, 라벨, 점수를 출력합니다.

## 빌드

이 디렉토리에서 다음 명령을 실행하세요.

```bash
cmake -B build -S .
cmake --build build -j
```

생성되는 바이너리:

- `build/infer-cls`

다음 명령으로 타겟 아키텍처를 확인할 수 있습니다.

```bash
file build/infer-cls
```

## 실행

샘플 이미지:

- `../rc/volcano.jpg`

### ARIES

```bash
./build/infer-cls ../../../compilation/image_classification/resnet50.mxq ../rc/volcano.jpg imagenet_labels.txt
```

### REGULUS

`build/infer-cls`, `resnet50.mxq`, `imagenet_labels.txt`, `volcano.jpg`를 타겟 보드로 복사한 뒤 다음 명령을 실행하세요.

```bash
chmod +x infer-cls
./infer-cls resnet50.mxq volcano.jpg imagenet_labels.txt
```

## 예상 출력

프로그램은 모델 입력 shape, 추론 시간, 상위 5개 ImageNet 예측 결과를 출력합니다.
