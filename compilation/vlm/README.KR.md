# Vision-Language 모델 컴파일

이 튜토리얼은 [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)의 인코더와 디코더를 MXQ로 컴파일하고, 실행에 필요한 파일을 하나의 모델 디렉터리로 준비합니다.

모든 명령은 `compilation/vlm`에서 실행합니다.

## 사전 준비

```bash
pip install -r requirements.txt
```

## 지원 디바이스

| 디바이스 | 지원 여부 |
| --- | --- |
| `aries-rb` | 지원 |
| `regulus-rb` | 지원 |
| `regulus-ra` | 미지원 |

## 1. 캘리브레이션 이미지 다운로드

```bash
python download_images.py
```

고정된 데이터셋 리비전에서 COCO 검증 이미지 300장을 내려받아 RGB로 변환하고 `224x224` 크기로 조정한 뒤
`./images`에 저장합니다.

## 2. 캘리브레이션 데이터 생성

```bash
python generate_calibration_data.py --batch-size 4
```

비전 인코더 데이터와 디코더의 prefill/decode 데이터를 `./calibration_data`에 생성합니다. 기본 배치 크기는 4이고
`cuda:0`을 사용하며, 24 GiB GPU에서 실행할 수 있도록 설정되어 있습니다. 다른 GPU는 `--device`로 지정합니다.

```text
calibration_data/
├── vision/
│   └── npy_files.txt
├── prefill/
│   └── npy_files.json
├── decode/
│   └── npy_files.json
└── language/
    └── npy_files.json
```

데이터셋 리비전, 난수 시드, 이미지 순서, 프롬프트 순서를 고정합니다. 같은 옵션, GPU, 소프트웨어 환경에서
반복 실행하면 동일한 캘리브레이션 파일을 생성합니다. EOS까지 생성된 결과만 캘리브레이션 데이터에 포함합니다.
`./calibration_data`가 이미 있으면 `--force`를 지정해 교체합니다.

## 3. MXQ 모델 컴파일

디코더를 먼저 컴파일합니다. 디코더 컴파일에서 인코더 컴파일과 런타임 모델 준비에 필요한 SpinR1 행렬을 생성합니다.

ARIES:

```bash
python compile_decoder.py --target-device aries-rb
python compile_encoder.py --target-device aries-rb
```

REGULUS:

```bash
python compile_decoder.py --target-device regulus-rb
python compile_encoder.py --target-device regulus-rb
```

각 스크립트는 대상 디바이스의 MBLT를 생성한 뒤 MXQ를 컴파일합니다. 두 스크립트의 컴파일 설정은 `compile_config.py`에 정의되어 있습니다.

```text
mblt/<target-device>/Qwen_Qwen3-VL-2B-Instruct_{decoder,encoder}.mblt
mxq/<target-device>/Qwen3-VL-2B-Instruct_{decoder,encoder}.mxq
spinWeight/<target-device>/global_rotation.pth
```

검증된 Qwen3-VL 2B 컴파일 설정은 자동으로 적용됩니다. ARIES는 `inference_scheme="all"`을 사용합니다. REGULUS는 `inference_scheme="single"`을 사용하며 최대 시퀀스 길이와 캐시 길이는 1024입니다.

## 4. 런타임 모델 준비

인코더와 디코더 MXQ 컴파일이 모두 끝난 뒤 실행합니다.

ARIES:

```bash
python prepare_model.py --target-device aries-rb
```

REGULUS:

```bash
python prepare_model.py --target-device regulus-rb
```

Mobilint 런타임 파일을 내려받고, 디코더 SpinR1 행렬을 토큰 임베딩에 적용하고, 두 MXQ와 디바이스 설정을 하나의 폴더에 구성합니다.

출력은 `./prepared/<target-device>/Qwen3-VL-2B-Instruct`에 저장됩니다. 해당 디렉터리가 이미 있으면 `--force`를 지정해 교체합니다.

## 런타임

[Python VLM 런타임 튜토리얼](../../runtime/python/vlm/README.KR.md)을 이어서 진행합니다.
