# 크로스 컴파일 런타임 (C++)

이 섹션은 호스트(x86_64)에서 C++ 추론 바이너리를 크로스 컴파일하고, 타겟 보드(ARM64)에서 실행하는 과정을 안내합니다.

## 사전 준비

### 1. 크로스 컴파일 툴체인 설치

[Mobilint 다운로드 센터](https://dl.mobilint.com/)에서 REGULUS -> Image Archive 메뉴의 최신 `tar.gz` 파일을 다운로드합니다.

압축을 해제하고 툴체인 설치 스크립트를 실행합니다:

```bash
tar -xzf {downloaded_tar_gz_file}
sudo ./install-regulus-toolchain.sh
```

설치가 완료되면 크로스 컴파일 환경을 활성화할 수 있습니다:

```bash
source /opt/crosstools/mobilint/{version}/environment-setup-cortexa53-mobilint-linux
```
