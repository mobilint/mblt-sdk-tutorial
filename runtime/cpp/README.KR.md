# C++ 런타임

C++ `qbruntime` 라이브러리는 ARIES 와 REGULUS 모두에서 동작합니다. 빌드 방식만 디바이스에 따라 달라집니다.

- **ARIES** (x86_64): 추론 바이너리를 호스트에서 네이티브로 빌드해 실행합니다.
- **REGULUS** (ARM64): x86_64 호스트에서 크로스 컴파일한 뒤 빌드된 바이너리를 타겟 보드로 배포해 실행합니다.

본 튜토리얼은 같은 `CMakeLists.txt` 로 두 흐름을 모두 다룹니다. x86_64 호스트에서 NPU 를 직접 쓰는 **ARIES 네이티브 빌드** 와, 호스트에서 크로스 컴파일해 타겟 보드로 배포하는 **REGULUS 크로스 컴파일** (툴체인 설치, CMake 크로스 빌드, 타겟 보드 배포).

> **참고**: 타겟 보드에는 Mobilint NPU 드라이버, 런타임 라이브러리, 유틸리티 도구가 미리 설치되어 있으므로 아래의 툴체인 설치와 크로스 빌드 단계만 진행하면 됩니다.

## 사전 준비 (REGULUS 전용)

아래 단계는 REGULUS 보드용 바이너리를 크로스 컴파일하기 위한 x86_64 호스트 환경 설정입니다. ARIES 네이티브 빌드에는 **필요하지 않으므로**, ARIES 만 사용하는 경우 이 섹션을 건너뜁니다.

### 1. 크로스 컴파일 툴체인 설치

[Mobilint 다운로드 센터](https://dl.mobilint.com/)에서 REGULUS -> Image Archive 메뉴의 최신 `tar.gz` 파일을 다운로드합니다.

압축을 해제하고 툴체인 설치 스크립트를 실행합니다:

```bash
tar -xzf {downloaded_tar_gz_file}
./install-regulus-toolchain.sh
```

설치가 완료되면 크로스 컴파일 환경을 활성화할 수 있습니다:

```bash
source /opt/crosstools/mobilint/{version}/environment-setup-cortexa53-mobilint-linux
```
