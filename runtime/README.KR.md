# Mobilint 런타임 튜토리얼

이 섹션의 튜토리얼은 Mobilint 런타임 라이브러리를 사용하여 모델을 실행하는 방법에 대한 안내를 제공합니다.

<div align="center">
<img src="../assets/Runtime.avif" width="75%", alt="Runtime Diagram">
</div>

## 런타임 준비

Mobilint qb 런타임 튜토리얼은 Mobilint NPU가 장착된 추론 PC에서 진행됩니다.

> 참고: 런타임 환경은 컴파일러 환경과 동일할 필요가 없습니다. 런타임은 Mobilint NPU가 장착된 시스템만을 요구합니다.

하드웨어 연결 후, 디바이스 접근을 활성화하기 위해 Mobilint NPU 드라이버를 시작해야 합니다.
자세한 지침은 [드라이버 설치 가이드](https://docs.mobilint.com/v0.29/en/installation.html#driver-installation)를 참조하세요.

드라이버가 성공적으로 설치되면 다음의 플래그를 사용하여 Docker 컨테이너 내에서 NPU 접근을 활성화할 수 있습니다:

```bash
--device /dev/aries0:/dev/aries0
```

다음으로, 런타임 라이브러리를 설치해야 합니다.
자세한 내용은 [런타임 라이브러리 설치 가이드](https://docs.mobilint.com/v0.29/en/installation.html#runtime-library-installtion)를 참조하세요.

Mobilint는 또한 NPU 상태 확인, MXQ 파일 검증, 간단한 추론 작업 실행을 위한 유틸리티 도구를 제공합니다.
자세한 내용은 [유틸리티 도구 설치 가이드](https://docs.mobilint.com/v0.29/en/installation.html#utility-installation)를 참조하세요.

이제 모델을 실행할 준비가 되었습니다!

현재 디렉토리의 튜토리얼을 따라 Mobilint NPU에서 컴파일된 모델을 실행해 보세요.
