# Python 런타임 (Python Runtime)

Python `qbruntime` 라이브러리는 ARIES 와 REGULUS 양쪽에서 동일한 NPU API 를 제공합니다. 드라이버와 런타임 라이브러리 설치의 일반 절차는 [런타임 개요](../README.KR.md) 를 참조하십시오. 본 문서는 Python 사용자에게 필요한 단계만 추립니다.

## 사용 절차

#### 1. 드라이버 활성화

호스트에서 Mobilint NPU 드라이버가 동작 중인지 확인합니다. 설치되지 않았다면 [드라이버 설치 가이드](https://docs.mobilint.com/v1.2/en/installing_driver.html) 를 따릅니다. Docker 환경에서는 컨테이너에 `--device /dev/aries0:/dev/aries0` 플래그로 NPU 를 노출합니다.

#### 2. Python 런타임 라이브러리 설치

```bash
pip install mobilint-qb-runtime
```

#### 3. 모델별 추가 의존성 설치

각 모델 튜토리얼 (`image_classification/`, `object_detection/`, `llm/`, `stt/`, ...) 에서 요구하는 패키지 (`numpy`, `PIL`, `torch`, `transformers` 등) 를 설치합니다. 정확한 목록은 해당 디렉토리 README 를 따릅니다.

#### 4. 스크립트 실행

원하는 모델 디렉토리로 이동해 추론 스크립트를 실행합니다.

## REGULUS 사전 설치

REGULUS 타겟 보드는 드라이버, `qbruntime` 라이브러리, 유틸리티 도구가 모두 출하 시점부터 설치되어 있습니다. 1, 2 단계를 건너뛰고 3 단계 (모델별 의존성) 부터 진행하면 됩니다.

## 디바이스 권장 사항

- **ARIES** (x86_64): **권장**. x86_64 호스트는 CPU 여유가 충분하고 Python 생태계가 완전히 갖춰져 있어 NPU 추론이 호스트의 전처리 또는 후처리에 병목되는 경우가 거의 없습니다.
- **REGULUS** (ARM64): **지원되지만 매우 느릴 수 있음**. Cortex-A53 호스트 CPU 는 일반적인 x86_64 호스트보다 훨씬 약하므로, NPU 추론 자체가 빠르더라도 Python 수준의 전처리·후처리·텐서 조작 (`numpy`, `torch`) 이 end-to-end 지연을 지배하는 경우가 많습니다. REGULUS 의 프로덕션 워크로드에는 [C++ 런타임](../cpp/README.KR.md) 을 권장합니다.
