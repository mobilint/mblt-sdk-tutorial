# Mobilint 런타임 튜토리얼 (Mobilint Runtime Tutorial)

Mobilint qb 런타임은 컴파일된 MXQ 모델을 Mobilint NPU에서 실행하기 위한 라이브러리입니다.
모델 로드, 추론 실행, 결과 반환까지의 과정을 처리합니다.

<!-- markdownlint-disable MD033 -->
<div align="center">
<img src="../assets/Runtime.avif" width="75%", alt="Runtime Diagram">
</div>
<!-- markdownlint-enable MD033 -->

런타임은 Python과 C++ 두 가지 언어로 제공됩니다.

## Python 런타임

Python `qbruntime` 라이브러리로 MXQ 모델을 실행한다. ARIES 와 REGULUS 는 동일한 API 를 제공하고, 환경 준비 단계만 디바이스에 따라 갈린다.

### ARIES (x86_64 호스트 + NPU)

호스트에 직접 모든 단계를 설치한다.

#### 1. 드라이버 설치

호스트의 Mobilint NPU 드라이버를 시작해 장치 액세스를 활성화한다. 자세한 내용은 [드라이버 설치 가이드](https://docs.mobilint.com/v1.2/en/installing_driver.html) 참고.

Docker 환경이면 컨테이너에 NPU 를 노출한다:

```bash
--device /dev/aries0:/dev/aries0
```

#### 2. 런타임 라이브러리 설치

Python 환경에 `qbruntime` 라이브러리를 설치한다 ([런타임 설치 가이드](https://docs.mobilint.com/v1.2/en/installing_runtime_library.html)):

```bash
pip install mobilint-qb-runtime
```

#### 3. 추가 종속성

모델에 따라 `torch`, `numpy`, `PIL`, `transformers` 등 Python 패키지가 추가로 필요할 수 있다. 필요 항목은 각 모델 튜토리얼을 따른다.

#### 4. 유틸리티 도구 (선택)

Mobilint 는 NPU 상태 확인, MXQ 검증, 간단한 추론 실행용 CLI 유틸리티를 제공한다. [유틸리티 도구 설치 가이드](https://docs.mobilint.com/v1.2/en/installing_utility.html) 참고.

### REGULUS (ARM64 타겟 보드)

REGULUS 보드에는 드라이버, `qbruntime` Python 라이브러리, CLI 유틸리티가 **미리 설치**되어 있다. 위 1, 2, 4 단계는 건너뛴다. 사용자가 보드에서 직접 해야 하는 것은 3 단계 (모델별 Python 추가 종속성 설치) 뿐이다.

---

## C++ 런타임

C++ `qbruntime` 라이브러리는 ARIES 와 REGULUS 모두에서 동작한다. 빌드 방식만 디바이스에 따라 달라진다.

- **ARIES** (x86_64): 추론 바이너리를 호스트에서 네이티브로 빌드해 실행한다.
- **REGULUS** (ARM64): x86_64 호스트에서 크로스 컴파일한 뒤 바이너리를 타겟 보드에 배포해 실행한다.

[`cpp/`](cpp/README.KR.md) 디렉토리는 같은 `CMakeLists.txt` 로 두 흐름을 모두 다룬다. ARIES 네이티브 빌드와 REGULUS 크로스 컴파일 (툴체인 설치, CMake 크로스 빌드, 타겟 보드 배포).
