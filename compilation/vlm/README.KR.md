# Vision-Language 모델 컴파일

이 튜토리얼은 [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)의 인코더와 디코더를 MXQ로 컴파일합니다.
실행에 필요한 파일은 하나의 모델 디렉터리로 준비합니다.

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

## 비전 모드

인코더는 두 가지 방식으로 만들 수 있습니다.

- **Static (기본값)** — 비전 그래프가 224x224를 position embedding과 rope 테이블에 상수로 굳혀 두므로, 런타임이 받는 모든 이미지는 먼저 224x224로 리사이즈됩니다.
  아래 모든 단계의 기본 동작입니다.
- **Dynamic** — 호스트가 전처리된 이미지 크기에 맞춰 위치 임베딩과 RoPE를 계산해 인코더 MXQ에 전달합니다.
  전처리기의 크기 조정 후에도 이미지 크기는 가변입니다.
  MBLT와 MXQ 파일에는 `_dynamic`, prepared 폴더에는 `-dynamic` 접미사가 붙습니다.

Dynamic 모드에서는 인코더와 디코더를 모두 `--dynamic`으로 컴파일해야 합니다.
디코더는 가변 이미지 토큰 수에 필요한 런타임 rope 입력을 노출합니다.
아래 모든 단계에 `--dynamic`을 넘기십시오.

## 1. 캘리브레이션 이미지 다운로드

```bash
python download_images.py
```

고정된 데이터셋 리비전에서 COCO 검증 이미지 300장을 내려받아 RGB로 변환하고 `224x224` 크기로 조정한 뒤 `./images`에 저장합니다.

Dynamic 비전에서는 리사이즈를 건너뛰어 샘플이 다양한 patch 수 N을 갖도록 합니다.

```bash
python download_images.py --dynamic
```

## 2. 캘리브레이션 데이터 생성

```bash
python generate_calibration_data.py --batch-size 4
```

비전 인코더 데이터와 디코더의 prefill/decode 데이터를 `./calibration_data`에 생성합니다.
기본 배치 크기는 4이며 `cuda:0`을 사용합니다.
사용 중인 GPU 메모리에 맞춰 `--batch-size`를 조절하고, 다른 GPU를 사용하려면 `--device`로 지정합니다.

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

각 비전 샘플은 `[1024, 64, 6]` 크기의 `images.npy`를 포함합니다.
각 디코더 샘플은 `[1, T, 2048]` 크기의 `inputs_embeds.npy`를 포함합니다.
DeepStack 3개는 `[3, T, 2048]` 크기의 `deepstack_visual_embeds.npy` 하나로 묶습니다.

Dynamic 비전에서는 `--dynamic`을 붙여 모든 샘플 작성기를 전환합니다.

```bash
python generate_calibration_data.py --batch-size 4 --dynamic
```

비전 샘플은 3-input이 됩니다.
입력은 folded pixel values `[1, 1, N, 1536]`, `pos_embeds` `[1, 1, N, 1024]`, packed rope `[1, 1, N, 128]`입니다.
매니페스트 `npy_files.json`은 N축이 dynamic으로 표시됩니다.
디코더 샘플에는 런타임 rope 슬롯에 대응하는 `cos.npy` `[1, T, 256]`이 세 번째 입력으로 추가됩니다.

데이터셋 리비전, 난수 시드, 이미지 순서, 프롬프트 순서를 고정합니다.
같은 옵션, GPU, 소프트웨어 환경에서 반복 실행하면 동일한 캘리브레이션 파일을 생성합니다.
EOS까지 생성된 결과만 캘리브레이션 데이터에 포함합니다.
`./calibration_data`가 이미 있으면 `--force`를 지정해 교체합니다.

## 3. MXQ 모델 컴파일

디코더를 먼저 컴파일합니다.
디코더 컴파일에서 인코더 컴파일과 런타임 모델 준비에 필요한 SpinR1 행렬을 생성합니다.

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

각 스크립트는 대상 디바이스의 MBLT를 생성한 뒤 MXQ를 컴파일합니다.
두 스크립트의 컴파일 설정은 `compile_config.py`에 정의되어 있습니다.

```text
mblt/<target-device>/Qwen_Qwen3-VL-2B-Instruct_{decoder,encoder}.mblt
mxq/<target-device>/Qwen3-VL-2B-Instruct_{decoder,encoder}.mxq
spinWeight/<target-device>/Qwen3-VL-2B-Instruct/global_rotation.pth
```

`--dynamic`을 붙이면 static 산출물 옆에 `_dynamic` 접미사가 붙은 짝이 생성됩니다.
MXQ 파일 이름은 `Qwen3-VL-2B-Instruct_{decoder,encoder}_dynamic.mxq`입니다.
SpinR1 행렬은 `spinWeight/<target-device>/Qwen3-VL-2B-Instruct-dynamic/global_rotation.pth`에 저장됩니다.
SpinR1 행렬 경로는 `(target-device, model-name, mode)` 단위로 분리되므로 같은 디바이스에서 여러 `--model-id`를 컴파일해도 서로 덮어쓰지 않습니다.

Qwen3-VL 2B 컴파일 설정은 자동으로 적용됩니다.
ARIES는 static과 dynamic 모두 `inference_scheme="all"`을 사용합니다.
REGULUS는 `inference_scheme="single"`을 사용하며 최대 시퀀스 길이와 캐시 길이는 1024입니다.

Dynamic 비전에서는 두 스크립트 모두에 `--dynamic`을 넘깁니다 (디코더 먼저).

```bash
python compile_decoder.py --target-device aries-rb --dynamic
python compile_encoder.py --target-device aries-rb --dynamic
```

Dynamic 모드에서는 호스트가 이미지와 텍스트를 합친 시퀀스의 RoPE를 계산해 임베딩, DeepStack 특징과 함께 디코더 MXQ에 전달합니다.
인코더 MBLT는 픽셀 데이터, 위치 임베딩, cosine, sine을 입력받으며 패치 수가 가변입니다.
컴파일 과정에서 cosine과 sine을 하나의 RoPE 입력으로 묶으므로 인코더 MXQ의 입력은 3개입니다.
Static과 dynamic 모두 현재 `qbcompiler.model_dict` 파서를 사용합니다.

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

출력은 `./prepared/<target-device>/Qwen3-VL-2B-Instruct`에 저장됩니다.
해당 디렉터리가 이미 있으면 `--force`를 지정해 교체합니다.

Dynamic 비전은 다음과 같이 준비합니다.

```bash
python prepare_model.py --target-device aries-rb --dynamic
```

`_dynamic` MXQ 쌍을 사용합니다.
`visual.pos_embed.weight`는 `model.safetensors`에 추가로 번들링합니다.
Dynamic 런타임 경로에서만 이 서브모듈을 할당합니다.
`config.json`의 최상위에는 `dynamic_vision=true`를 씁니다.
출력은 `./prepared/<target-device>/Qwen3-VL-2B-Instruct-dynamic`에 저장됩니다.

## 다른 모델 크기

컴파일 · 캘리브레이션 · 준비 스크립트 모두 `--model-id`를 받습니다.
기본값은 `Qwen/Qwen3-VL-2B-Instruct`입니다.
`Qwen/Qwen3-VL-4B-Instruct`나 `Qwen/Qwen3-VL-8B-Instruct` 같은 다른 id도 같은 파이프라인에서 사용할 수 있습니다.
런타임 템플릿 레포지토리 id는 `mobilint/<name>`으로 유도되며 Mobilint가 `mobilint/Qwen3-VL-{2B,4B,8B}-Instruct`를 공개합니다.

`compile_config.py`의 컴파일 설정은 2B 기준으로 조정되어 있습니다.
다른 모델 크기는 `ENCODER_16BIT_ACTIVATIONS`의 레이어 이름 확인을 포함해 컴파일과 추론을 별도로 검증해야 합니다.

## 런타임

[Python VLM 런타임 튜토리얼](../../runtime/python/vlm/README.KR.md)을 이어서 진행합니다.
런타임 스크립트의 기본 `--model-folder`는 static 2B 준비 폴더를 가리키므로, dynamic 빌드나 2B가 아닌 `--model-id`를 사용했다면 실제 폴더 경로를 명시적으로 넘기십시오.

```bash
# Dynamic 2B
python ../../runtime/python/vlm/inference_mblt_model_zoo.py \
    --model-folder prepared/aries-rb/Qwen3-VL-2B-Instruct-dynamic

# Static 4B (또는 8B): `-dynamic` 접미사 제거
python ../../runtime/python/vlm/inference_mblt_model_zoo.py \
    --model-folder prepared/aries-rb/Qwen3-VL-4B-Instruct
```
