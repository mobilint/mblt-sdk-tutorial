# mobilint-cli 설치

[English](install.md) | 한국어

Python이나 가상환경은 필요하지 않습니다.

Windows 사용자는 [공식 Windows 설치 안내](https://docs.mobilint.com/runtime/en/v1.6/installing_utility/#option-4-installation-via-the-installer-windows)를 참고하세요.

## 실행 환경 준비

- REGULUS EVK: 보드에 `mobilint-cli`가 미리 설치되어 있습니다.
  패키지를 업데이트하려면 개별 패키지만 갱신하기보다 [Mobilint 다운로드 센터](https://dl.mobilint.com/)에서 해당 보드용 최신 `Image Archive`를 내려받아 보드 이미지 전체를 업데이트하는 것을 권장합니다.
- Ubuntu 호스트: NPU를 연결한 호스트에 해당 장치의 드라이버를 먼저 설치하세요.
  ARIES는 [공식 드라이버 설치 안내](https://docs.mobilint.com/aries/en/latest/driver-installation/)를 따르세요.
- Docker: 드라이버는 컨테이너가 아닌 호스트에 설치하세요.
  ARIES의 첫 장치는 호스트에서 `ls -l /dev/aries0`로 확인합니다.
  컨테이너를 생성할 때 `docker run`에 `--device=/dev/aries0`를 추가하고, 컨테이너 안에서도 같은 경로가 보이는지 확인하세요.
- Ubuntu 설치 명령은 CLI를 사용할 환경 안에서 실행하세요.
  `root`로 실행하는 경우에는 `sudo`를 생략하세요.

## Ubuntu: Mobilint APT 저장소 등록 및 설치

CLI를 설치할 환경에서 실행하세요. 작업 디렉터리는 관계없습니다.

드라이버나 런타임을 설치하면서 같은 환경에 Mobilint APT 저장소를 이미 등록했다면 세 번째 단계부터 실행하세요.

1. HTTPS 다운로드에 필요한 패키지를 설치합니다.

   ```bash
   sudo apt-get update
   sudo apt-get install ca-certificates curl
   ```

2. Mobilint 서명 키를 내려받고, 이 키를 사용할 저장소를 등록합니다.
   `deb`로 시작하는 저장소 항목은 한 줄로 작성하세요.

   ```bash
   sudo install -d -m 0755 /etc/apt/keyrings
   sudo curl -fsSL https://dl.mobilint.com/apt/gpg.pub -o /etc/apt/keyrings/mblt.asc
   sudo chmod a+r /etc/apt/keyrings/mblt.asc
   echo 'deb [signed-by=/etc/apt/keyrings/mblt.asc] https://dl.mobilint.com/apt stable multiverse' | sudo tee /etc/apt/sources.list.d/mobilint.list
   ```

3. 패키지 목록을 갱신하고 CLI를 설치합니다.

   ```bash
   sudo apt-get update
   sudo apt-get install mobilint-cli
   ```

다른 Linux 설치 방식은 [공식 유틸리티 설치 안내](https://docs.mobilint.com/runtime/en/v1.6/installing_utility/#install-mobilint-cli)를 참고하세요.

## 설치 확인

```bash
mobilint-cli --version
```

CLI 버전이 출력되면 설치가 완료된 것입니다.

NPU 연결은 [사용법](usage.KR.md)의 `devices` 명령으로 확인하세요.
