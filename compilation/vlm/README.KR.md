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

## 비전 인코더

인코더는 qbcompiler의 `vision` 모델 파트를 dynamic side input과 함께 사용해 컴파일합니다.
호스트가 전처리된 이미지 크기에 맞춰 비전 위치 임베딩과 RoPE를 계산해 인코더 MXQ에 전달하므로, 전처리기의 크기 조정 후에도 이미지 크기는 가변이며 이미지 토큰 수는 이미지 크기를 따릅니다.
디코더는 런타임 RoPE 입력(`dynamic_rope`)을 갖도록 컴파일되어 가변 이미지 토큰 수를 받습니다.

> **qbcompiler 1.4 이상이 필요합니다.**
> 이전 버전의 static 224x224 인코더(`qbcompiler.model_dict_legacy`와 `repreprocess_pixel_values` 사용)는 레거시 파서와 함께 제거되었습니다.
> 2B 모델 기준 인코더 MXQ의 입력은 folded pixel values `[1, N, 1536]`, 위치 임베딩 `[1, N, 1024]`, packed RoPE `[1, N, 128]` 3개입니다.
> `mblt-model-zoo` 2.11 이상은 입력 shape로 이 인코더를 인식합니다.

## 1. 캘리브레이션 이미지 다운로드

```bash
python download_images.py
```

고정된 데이터셋 리비전에서 COCO 검증 이미지 300장을 내려받아 RGB로 변환하고 원본 해상도 그대로 `./images`에 저장합니다.
원본 크기를 유지하므로 캘리브레이션 샘플이 다양한 비전 patch 수 N을 갖습니다.

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
│   └── npy_files.json
├── prefill/
│   └── npy_files.json
├── decode/
│   └── npy_files.json
└── language/
    └── npy_files.json
```

각 비전 샘플은 3-input입니다.
입력은 folded pixel values `[1, 1, N, 1536]`, `pos_embeds` `[1, 1, N, 1024]`, packed RoPE `[1, 1, N, 128]`입니다.
매니페스트 `npy_files.json`은 N축이 dynamic으로 표시됩니다.
각 디코더 샘플은 `inputs_embeds.npy`, 분리된 DeepStack 파일 `deepstack_0.npy`, `deepstack_1.npy`, `deepstack_2.npy`, 그리고 RoPE 텐서 `cos.npy`를 포함합니다.
임베딩과 DeepStack 파일의 크기는 `[1, 1, T, 2048]`이고, `cos.npy`는 `[1, 1, T, 256]` 크기로 디코더의 런타임 RoPE 입력에 대응합니다.

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

각 스크립트는 `load_for_part`로 모델을 불러오고, `capture_forward_inputs`로 한 번의 forward 입력을 캡처한 뒤, `mblt_compile(model_part=...)`로 대상 디바이스의 MBLT를 생성하고 MXQ를 컴파일합니다.
디코더는 `language` 파트를 사용합니다.
인코더는 `vision` 파트를 `model_part_options={"side_inputs": True}`와 함께 사용합니다. 이 옵션은 위치 임베딩, cosine, sine을 folded pixel values와 함께 그래프 입력으로 만들고, 스크립트는 이들의 patch 수 축을 dynamic으로 지정합니다.
컴파일 과정에서 cosine과 sine을 하나의 RoPE 입력으로 묶으므로 인코더 MXQ의 입력은 3개입니다.
두 스크립트의 컴파일 설정은 `compile_config.py`에 정의되어 있습니다.

```text
mblt/<target-device>/Qwen_Qwen3-VL-2B-Instruct_{decoder,encoder}.mblt
mxq/<target-device>/Qwen3-VL-2B-Instruct_{decoder,encoder}.mxq
spinWeight/<target-device>/Qwen3-VL-2B-Instruct/global_rotation.pth
```

SpinR1 행렬 경로는 `(target-device, model-name)` 단위로 분리되므로 같은 디바이스에서 여러 `--model-id`를 컴파일해도 서로 덮어쓰지 않습니다.

Qwen3-VL 2B 컴파일 설정은 자동으로 적용됩니다.
ARIES는 `inference_scheme="all"`을 사용합니다.
REGULUS는 `inference_scheme="single"`을 사용하며 최대 시퀀스 길이와 캐시 길이는 4096입니다.

qbcompiler 1.4는 bias correction(`BiasCorrectionConfig`)을 기본으로 켜며, `compile_config.py`는 이 기본값을 그대로 사용합니다.
따라서 MXQ 결과와 컴파일 시간이 qbcompiler 1.3으로 만든 빌드와 다릅니다.
가중치 양자화 중 GPU 메모리 사용량은 자동으로 제한됩니다(`ResourceManagementConfig.gpu_memory_budget_mb`, 기본값 `-1`). 다른 작업과 GPU를 함께 쓴다면 MiB 단위로 예산을 지정하십시오.

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

Mobilint 런타임 파일을 내려받고, 디코더 SpinR1 행렬을 토큰 임베딩에 적용하고, 호스트 측 위치 임베딩 계산에 쓰이는 `visual.pos_embed.weight`를 함께 담고, 두 MXQ와 디바이스 설정을 하나의 폴더에 구성합니다.
`config.json`의 최상위에는 `dynamic_vision=true`를 씁니다.

출력은 `./prepared/<target-device>/Qwen3-VL-2B-Instruct`에 저장됩니다.
해당 디렉터리가 이미 있으면 `--force`를 지정해 교체합니다.

## 출력 구조

ARIES 빌드를 컴파일하고 준비하면 생성 파일은 다음과 같습니다.

```text
images/

calibration_data/
├── vision/
├── prefill/
├── decode/
└── language/

mblt/aries-rb/
├── Qwen_Qwen3-VL-2B-Instruct_decoder.mblt
└── Qwen_Qwen3-VL-2B-Instruct_encoder.mblt

mxq/aries-rb/
├── Qwen3-VL-2B-Instruct_decoder.mxq
└── Qwen3-VL-2B-Instruct_encoder.mxq

spinWeight/aries-rb/
└── Qwen3-VL-2B-Instruct/

prepared/aries-rb/
└── Qwen3-VL-2B-Instruct/
```

## 다른 모델 크기

컴파일 · 캘리브레이션 · 준비 스크립트 모두 `--model-id`를 받습니다.
기본값은 `Qwen/Qwen3-VL-2B-Instruct`입니다.
`Qwen/Qwen3-VL-4B-Instruct`나 `Qwen/Qwen3-VL-8B-Instruct` 같은 다른 id도 같은 파이프라인에서 사용할 수 있습니다.
런타임 템플릿 레포지토리 id는 `mobilint/<name>`으로 유도되며 Mobilint가 `mobilint/Qwen3-VL-{2B,4B,8B}-Instruct`를 공개합니다.

`compile_config.py`의 컴파일 설정은 2B 기준으로 조정되어 있습니다.
다른 모델 크기는 컴파일과 추론을 별도로 검증해야 합니다.
16-bit activation 레이어(decoder graph 입력, encoder graph 출력)는 `compile_config.py`가 MBLT에서 읽으므로 모델 크기에 따라 자동으로 정해집니다.

## 런타임

[Python VLM 런타임 튜토리얼](../../runtime/python/vlm/README.KR.md)을 이어서 진행합니다.
런타임 스크립트의 기본 `--model-folder`는 ARIES 2B 준비 폴더를 가리키므로, 다른 대상 디바이스나 2B가 아닌 `--model-id`를 사용했다면 실제 폴더 경로를 명시적으로 넘기십시오.

```bash
python ../../runtime/python/vlm/inference_mblt_model_zoo.py \
    --model-folder prepared/aries-rb/Qwen3-VL-4B-Instruct
```
