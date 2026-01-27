# Mobilint 런타임 튜토리얼 (Mobilint Runtime Tutorial)

이 섹션은 Mobilint 런타임 라이브러리를 사용하여 모델을 실행하는 방법에 대한 자세한 지침을 제공합니다.

<div align="center">
<img src="../assets/Runtime.avif" width="75%", alt="Runtime Diagram">
</div>

## 런타임 준비 (Runtime Preparation)

Mobilint `qbruntime` 튜토리얼은 Mobilint NPU가 장착된 추론 PC에서 작업한다고 가정합니다.

> **참고**: 런타임 환경은 컴파일 환경과 동일할 필요는 없습니다. 런타임은 Mobilint NPU가 장착된 시스템만 있으면 됩니다.

### 1. 드라이버 설치 (Driver Installation)

하드웨어 연결 후, Mobilint NPU 드라이버를 시작하여 장치 액세스를 활성화하십시오.
자세한 지침은 [드라이버 설치 가이드](https://docs.mobilint.com/v0.29/en/installation.html#driver-installation)에서 확인할 수 있습니다.

드라이버가 성공적으로 설치되었고 Docker를 사용하는 경우, 다음 플래그를 사용하여 컨테이너 내부에서 NPU 액세스를 활성화할 수 있습니다:

```bash
--device /dev/aries0:/dev/aries0
```

### 2. 런타임 라이브러리 설치 (Runtime Library Installation)

다음으로 런타임 라이브러리를 설치하십시오.
자세한 내용은 [런타임 설치 가이드](https://docs.mobilint.com/v0.29/en/installation.html#runtime-library-installtion)를 참조하십시오.

Python 환경에서 런타임 라이브러리를 설치하려면 다음 명령어를 사용하십시오:

```bash
pip install qbruntime
```

### 3. 추가 종속성 (Additional Dependencies)

모델 유형에 따라 추가 Python 패키지(예: `torch`, `numpy`, `PIL`, `transformers`)가 필요할 수 있습니다. 자세한 요구 사항은 각 모델별 튜토리얼을 참조하십시오.

### 4. 유틸리티 도구 (선택 사항)

Mobilint는 NPU 상태 확인, MXQ 파일 검증, 간단한 추론 작업 실행을 위한 유틸리티 도구도 제공합니다.
자세한 내용은 [유틸리티 도구 설치 가이드](https://docs.mobilint.com/v0.29/en/installation.html#utility-installation)를 참조하십시오.

---

## 실행 준비 완료 (Ready to Run?)

이제 모델을 실행할 준비가 되었습니다!
이 디렉토리의 튜토리얼을 통해 Mobilint NPU에서 컴파일된 모델을 실행해 보십시오.
