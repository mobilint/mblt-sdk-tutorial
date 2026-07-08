# Python 런타임

Python `qbruntime` 라이브러리는 ARIES와 REGULUS에서 동일한 Mobilint NPU API를 제공합니다. 이 문서는 이 디렉토리의 튜토리얼을 실행하기 위한 Python 전용 설정만 다룹니다. 전체 런타임 설치 및 디바이스 설정은 [런타임 개요](../README.KR.md)를 참고하세요.

## 빠른 시작

이 디렉토리의 Python 튜토리얼을 실행하기 전에 아래 단계를 먼저 수행하세요.

### 1. NPU 드라이버 활성화

호스트에 Mobilint NPU 드라이버가 설치되어 있고 정상적으로 실행 중인지 확인하세요. 아직 설치하지 않았다면 [드라이버 설치 가이드](https://docs.mobilint.com/v1.2/en/installing_driver.html)를 따르세요.

Docker 환경에서는 다음 옵션으로 디바이스를 컨테이너에 노출해야 합니다.

```bash
--device /dev/aries0:/dev/aries0
```

### 2. Python 런타임 라이브러리 설치

```bash
pip install mobilint-qb-runtime
```

### 3. 튜토리얼별 의존성 설치

각 튜토리얼 디렉토리에는 필요한 Python 패키지가 별도로 정리되어 있습니다. 모델에 따라 `numpy`, `Pillow`, `torch`, `transformers`, `mblt-model-zoo` 같은 패키지가 필요할 수 있습니다.

실행하려는 튜토리얼의 README에 적힌 의존성을 설치하세요. 예:

- `image_classification/`
- `object_detection/`
- `bert/`
- `llm/`
- `stt/`
- `vlm/`

### 4. 튜토리얼 스크립트 실행

원하는 튜토리얼 디렉토리로 이동한 뒤, 해당 README에 안내된 스크립트를 그 디렉토리에서 실행하세요.

## REGULUS 사전 설치 환경

REGULUS 타겟 보드에는 드라이버, `qbruntime`, 유틸리티 도구가 미리 설치되어 있는 경우가 많습니다. 이런 환경에서는 보통 1단계와 2단계를 건너뛰고 튜토리얼별 의존성 설치부터 진행하면 됩니다.

## 디바이스 권장 사항

- **ARIES** (`x86_64`): 권장. 호스트 측 전처리와 후처리가 병목이 될 가능성이 상대적으로 낮습니다.
- **REGULUS** (`ARM64`): 지원되지만 Python 워크로드는 느릴 수 있습니다. 전처리, 후처리, 텐서 조작이 전체 지연 시간의 대부분을 차지할 수 있기 때문입니다.

REGULUS에서 프로덕션 성격의 워크로드를 실행해야 한다면 가능하면 [C++ 런타임](../cpp/README.KR.md)을 사용하는 것이 좋습니다.
