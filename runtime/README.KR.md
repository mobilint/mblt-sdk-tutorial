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

Python `qbruntime` 라이브러리를 사용하여 MXQ 모델을 실행합니다. ARIES 와 REGULUS 모두 동일한 API 를 사용합니다.

### 런타임 준비

Mobilint `qbruntime` 튜토리얼은 Mobilint NPU 가 장착된 시스템 또는 타겟 보드에서 작업한다고 가정합니다.

> **참고**: 런타임 환경은 컴파일 환경과 동일할 필요는 없습니다. Mobilint NPU 가 장착된 시스템 (예: ARIES) 또는 타겟 보드 (예: REGULUS) 면 됩니다. 타겟 보드의 경우 드라이버, 런타임 라이브러리, 유틸리티 도구가 모두 미리 설치되어 있어 아래 해당 설치 단계들을 생략할 수 있습니다.

#### 1. 드라이버 설치

하드웨어 연결 후, Mobilint NPU 드라이버를 시작하여 장치 액세스를 활성화하십시오.
자세한 지침은 [드라이버 설치 가이드](https://docs.mobilint.com/v0.29/en/installation.html#driver-installation)에서 확인할 수 있습니다.

드라이버가 성공적으로 설치되었고 Docker를 사용하는 경우, 다음 플래그를 사용하여 컨테이너 내부에서 NPU 액세스를 활성화할 수 있습니다:

```bash
--device /dev/aries0:/dev/aries0
```

#### 2. 런타임 라이브러리 설치

다음으로 런타임 라이브러리를 설치하십시오.
자세한 내용은 [런타임 설치 가이드](https://docs.mobilint.com/v0.29/en/installation.html#runtime-library-installtion)를 참조하십시오.

Python 환경에서 런타임 라이브러리를 설치하려면 다음 명령어를 사용하십시오:

```bash
pip install mobilint-qb-runtime
```

#### 3. 추가 종속성

모델 유형에 따라 추가 Python 패키지(예: `torch`, `numpy`, `PIL`, `transformers`)가 필요할 수 있습니다. 자세한 요구 사항은 각 모델별 튜토리얼을 참조하십시오.

#### 4. 유틸리티 도구 (선택 사항)

Mobilint는 NPU 상태 확인, MXQ 파일 검증, 간단한 추론 작업 실행을 위한 유틸리티 도구도 제공합니다.
자세한 내용은 [유틸리티 도구 설치 가이드](https://docs.mobilint.com/v0.29/en/installation.html#utility-installation)를 참조하십시오.

---

## C++ 런타임

C++ `qbruntime` 라이브러리는 ARIES 와 REGULUS 모두에서 동작합니다. 빌드 방식만 디바이스에 따라 달라집니다.

- **ARIES** (x86_64): 추론 바이너리를 호스트에서 네이티브로 빌드해 실행합니다.
- **REGULUS** (ARM64): x86_64 호스트에서 크로스 컴파일한 뒤 빌드된 바이너리를 타겟 보드에 배포해 실행합니다.

[`cpp/`](cpp/README.md) 디렉토리는 현재 REGULUS 크로스 컴파일 흐름 (툴체인 설치, CMake 크로스 빌드, 타겟 보드 배포) 을 다룹니다.
