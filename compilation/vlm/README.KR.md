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

## 비전 모드

인코더는 두 가지 방식으로 만들 수 있습니다.

- **Static (기본값)** — 비전 그래프가 224x224를 position embedding과 rope 테이블에 상수로 굳혀 두므로, 런타임이 받는 모든 이미지는 먼저 224x224로 리사이즈됩니다. 아래 모든 단계의 기본 동작입니다.
- **Dynamic** — position embedding과 rope 테이블이 그래프 입력이 되어 런타임이 원본 해상도 이미지를 그대로 인코더에 넣습니다. 산출 파일에는 `_dynamic` 접미사가 붙어 `mblt/`, `mxq/`, `prepared/` 아래에 두 변종을 나란히 둘 수 있습니다.

Dynamic 비전은 *bundled release*입니다. 텍스트 디코더도 런타임 rope 입력을 노출해야 하며 (`compile_decoder.py --dynamic`), `mblt-model-zoo`가 짝이 안 맞으면 로드를 거부합니다. Dynamic 경로로 가려면 아래 모든 단계에 `--dynamic`을 넘기십시오.

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

비전 인코더 데이터와 디코더의 prefill/decode 데이터를 `./calibration_data`에 생성합니다. 기본 배치 크기는 4이며 `cuda:0`을 사용합니다. 사용 중인 GPU 메모리에 맞춰 `--batch-size`를 조절하고, 다른 GPU를 사용하려면 `--device`로 지정합니다.

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

각 비전 샘플은 `[1024, 64, 6]` 크기의 `images.npy`를 포함합니다. 각 디코더 샘플은 `[1, T, 2048]` 크기의 `inputs_embeds.npy`와 DeepStack 3개를 하나로 묶은 `[3, T, 2048]` 크기의 `deepstack_visual_embeds.npy`를 포함합니다.

Dynamic 비전에서는 `--dynamic`을 붙여 모든 샘플 작성기를 전환합니다.

```bash
python generate_calibration_data.py --batch-size 4 --dynamic
```

비전 샘플은 3-input이 됩니다 (folded pixel values `[1, 1, N, 1536]`, `pos_embeds` `[1, 1, N, 1024]`, packed rope `[1, 1, N, 128]`). 매니페스트 `npy_files.json`은 N축이 dynamic으로 표시됩니다. 디코더 샘플에는 런타임 rope 슬롯에 대응하는 `cos.npy` `[1, T, 256]`이 세 번째 입력으로 추가됩니다.

데이터셋 리비전, 난수 시드, 이미지 순서, 프롬프트 순서를 고정합니다. 같은 옵션, GPU, 소프트웨어 환경에서 반복 실행하면 동일한 캘리브레이션 파일을 생성합니다. EOS까지 생성된 결과만 캘리브레이션 데이터에 포함합니다. `./calibration_data`가 이미 있으면 `--force`를 지정해 교체합니다.

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

Dynamic 비전에서는 두 스크립트 모두에 `--dynamic`을 넘깁니다 (디코더 먼저).

```bash
python compile_decoder.py --target-device aries-rb --dynamic
python compile_encoder.py --target-device aries-rb --dynamic
```

디코더는 `LlmConfig.attributes.runtime.dynamic_rope=True`를 통해 cos/sin `InputConstant`를 런타임 rope 입력으로 승격시킵니다. 인코더는 V2 dispatch(`qbcompiler.model_dict`)로 전환되어 `pos_embeds`, `cos`, `sin`을 그래프 입력으로 다루고 N축을 dynamic으로 표시합니다. 산출 파일은 `mxq/<target-device>/Qwen3-VL-2B-Instruct_{decoder,encoder}_dynamic.mxq`에 저장됩니다.

`compile_config.py`의 `ENCODER_16BIT_ACTIVATIONS`는 static Qwen3-VL-2B 빌드에서 관측한 그래프 유래 연산자 이름 목록입니다. Dynamic 파싱이나 다른 모델 크기에서는 이 이름이 달라질 수 있습니다. 일치하지 않는 항목은 양자화기가 조용히 무시하므로 목록이 틀려도 비전 SQNR이 조금 떨어질 뿐 컴파일은 실패하지 않습니다.

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

Dynamic 비전은 다음과 같이 준비합니다.

```bash
python prepare_model.py --target-device aries-rb --dynamic
```

`_dynamic` MXQ 쌍을 집어오고, `visual.pos_embed.weight`를 `model.safetensors`에 추가로 번들링하며 (dynamic 런타임 경로만 이 서브모듈을 할당합니다), `config.json`의 최상위에 `dynamic_vision=true`를 씁니다. 출력은 `./prepared/<target-device>/Qwen3-VL-2B-Instruct-dynamic`에 저장됩니다.

## 다른 모델 크기

컴파일 · 캘리브레이션 · 준비 스크립트 모두 `--model-id`를 받습니다. 기본값은 `Qwen/Qwen3-VL-2B-Instruct`이고, `Qwen/Qwen3-VL-4B-Instruct`나 `Qwen/Qwen3-VL-8B-Instruct` 같은 다른 id를 넘기면 같은 파이프라인이 그 모델을 대상으로 돕니다. 런타임 템플릿 레포지토리 id는 `mobilint/<name>`으로 유도되며 Mobilint가 `mobilint/Qwen3-VL-{2B,4B,8B}-Instruct`를 공개합니다.

`compile_config.py`의 컴파일 설정은 2B 기준으로 조정되어 있습니다. 다른 크기에서도 컴파일은 성공하지만 `ENCODER_16BIT_ACTIVATIONS` 목록이 새 그래프와 안 맞을 수 있고, ARIES 브랜치의 `hessian_quant_config=None`은 2B에서만 검증되었습니다. 둘 다 품질 노브이지 정확성 제약은 아닙니다.

## 런타임

[Python VLM 런타임 튜토리얼](../../runtime/python/vlm/README.KR.md)을 이어서 진행합니다.
