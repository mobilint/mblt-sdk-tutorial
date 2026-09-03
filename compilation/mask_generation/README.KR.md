# 마스크 생성 모델 컴파일

이 튜토리얼은 Mobilint `qbcompiler`로 프롬프트 기반 분할(promptable segmentation) 모델을 컴파일하는 방법을 설명합니다.

이 예제에서는 Meta의 [SAM2 Hiera large](https://github.com/facebookresearch/sam2) 모델을 사용합니다. SAM2는 단일 순전파 네트워크가 아닙니다. 이미지 인코더는 이미지당 한 번 실행되고, 가벼운 마스크 디코더는 프롬프트당 한 번 실행됩니다. 이 튜토리얼에서는 두 모델을 각각 MXQ로 컴파일하고, 프롬프트 인코더는 호스트에 남겨 둡니다.

> **참고**: 기본 워크플로에서는 인코더와 디코더 모두 qbcompiler의 legacy torch parser로 SAM2에서 직접 MBLT로 내보냅니다. ONNX 내보내기나 ONNX 프론트엔드는 사용하지 않습니다.

## 사전 준비

시작하기 전에 다음 항목이 준비되어 있어야 합니다:

- 서로 맞는 조합의 `qbcompiler` 및 `mblt` Python 패키지. 두 내보내기 모두 qbcompiler의 legacy torch parser를 사용합니다.
- Python 3.10 이상
- 캘리브레이션용 CUDA GPU
- [facebookresearch/sam2](https://github.com/facebookresearch/sam2) 로컬 checkout
- SA-V 아카이브(`sav_train` 청크 또는 `sav_val.tar` / `sav_test.tar`). Meta에서 직접 받아 `prepare_sav.py`로 준비합니다. [단계 0](#단계-0-sa-v-캘리브레이션-소스-준비)을 참고하십시오.
- 파싱 wrapper가 고정하고 있는 `transformers==4.57.1`

필요한 Python 패키지는 다음과 같이 설치합니다:

```bash
pip install -r requirements.txt
```

SAM2는 PyPI에 없으므로 공식 저장소에서 설치합니다:

```bash
git clone https://github.com/facebookresearch/sam2.git /workspace/sam2
pip install -e /workspace/sam2
```

패키지 자체를 설치하지 않으려면 원하는 위치에 clone한 뒤 `--sam2-root`로 경로를 전달하십시오. 이 옵션은 checkout을 `sys.path`에 추가할 뿐이므로 SAM2 자체의 의존성은 별도로 설치해야 합니다. SAM2는 `requirements.txt`가 아니라 패키지 메타데이터로 의존성을 선언하므로 직접 설치하십시오:

```bash
pip install 'torch>=2.5.1' 'torchvision>=0.20.1' 'numpy>=1.24.4' 'pillow>=9.4.0' 'hydra-core>=1.3.2' 'iopath>=0.1.10' 'tqdm>=4.66.1'
```

설치하지 않으면 이 튜토리얼의 `requirements.txt`를 모두 만족하더라도 `sam2.sam2_image_predictor` import가 실패합니다.

SAM2 체크포인트는 최초 실행 시 Hugging Face에서 다운로드되므로, 캘리브레이션 호스트에는 네트워크 접근 또는 미리 준비된 Hugging Face 캐시가 필요합니다.

> **참고**: `qbcompiler`는 처리할 수 없는 SAM2 디코더 variant를 컴파일 시점에 거부합니다. 파싱 wrapper는 `use_multimask_token_for_obj_ptr=True`인 공식 Hiera 디코더 계약을 요구합니다.

## 개요

워크플로우는 크게 네 단계로 구성됩니다:

0. **캘리브레이션 소스 준비**: `prepare_sav.py`로 SA-V subset을 추출합니다.
1. **MBLT 그래프 생성**: 인코더와 디코더를 모두 legacy parser로 직접 파싱합니다.
2. **캘리브레이션 데이터셋 준비**: SA-V에서 인코더와 디코더 캘리브레이션 텐서를 생성합니다.
3. **모델 컴파일**: 해당 캘리브레이션 데이터로 두 MBLT 그래프를 `.mxq`로 변환합니다.

## SAM2 분할 구조

컴파일된 모델이 네트워크 전체를 담당하지는 않습니다. 다음 세 단계는 호스트에 남아 공식 SAM2 코드로 실행됩니다.

```text
image
  -> SAM2 이미지 변환                          호스트
  -> 이미지 인코더                             인코더 MXQ
  -> 프롬프트 인코더                           호스트
  -> 마스크 디코더                             디코더 MXQ
  -> 마스크 업스케일링                         호스트
```

디코더 MXQ는 프롬프트 인코더의 출력을 그대로 받습니다. 출력 토큰 concat과 `image_embeddings + dense_prompt_embeddings` 합은 파서가 만들어 주는 host bridge 서브그래프 안에서 처리됩니다. 그 입력은 여전히 호스트가 준비하므로, 캘리브레이션은 런타임 튜토리얼이 사용하는 것과 정확히 동일한 호스트 경로에서 생성해야 합니다. `sam2_host.py`가 그 공유 경로이며, 각 디렉토리를 독립적으로 유지하기 위해 런타임 튜토리얼에도 동일한 파일이 있습니다.

### 디코더 입력 순서

디코더에는 입력이 6개 있고 그중 3개가 `(1, 256, 64, 64)`로 같은 shape이므로, 배열 위치만으로는 입력을 구분할 수 없습니다. 따라서 캘리브레이션은 각 슬롯의 MBLT input name과 semantic role을 함께 기록합니다.

캘리브레이션과 컴파일에 사용하는 현재 MBLT input name 순서:

```text
image_embeddings            -> image_embeddings          (1, 256,  64,  64)
dense_prompt_embeddings     -> dense_prompt_embeddings   (1, 256,  64,  64)
image_pe                    -> image_pe                  (1, 256,  64,  64)
sparse_prompt_embeddings_0  -> sparse_prompt_embeddings  (1,   1,   N, 256)
high_res_features0_0        -> hrf0_nhwc                 (1, 256, 256,  32)
high_res_features1_0        -> hrf1_nhwc                 (1, 128, 128,  64)
```

`sparse_prompt_embeddings`가 프롬프트 축 `N`을 가지며, 포인트 하나당 임베딩 1개에 패딩 1개가 더해진 길이입니다. 디코더 MBLT의 input name이 다르다면 기본값이 그대로 유효하다고 가정하지 말고 `--decoder-input-bindings`로 새 매핑을 제공하십시오.

## 단계 0: SA-V 캘리브레이션 소스 준비

캘리브레이션 프레임과 마스크는 SA-V에서 가져옵니다. **먼저 직접 다운로드하십시오.** 공식 [SA-V 데이터셋 가이드](https://github.com/facebookresearch/sam2/blob/main/sav_dataset/README.md)가 각 split을 설명하고, 양식 동의가 필요한 [다운로드 페이지](https://ai.meta.com/datasets/segment-anything-video-downloads/)로 안내합니다. 이 튜토리얼은 데이터를 직접 받아오지 않으며 게이트를 우회하지도 않습니다. 미러가 아니라 Meta에서 직접 받으십시오.

명령을 실행하는 디렉터리에 `sav_val.tar`를 두십시오. `./sav_val.tar`가
기본값이므로, 다음 명령으로 범위가 제한된 캘리브레이션 subset을 추출합니다:

```bash
python prepare_sav.py
```

두 가지 SA-V 레이아웃을 모두 지원하며 자동으로 감지합니다:

| Split | 레이아웃 | 파일 |
| --- | --- | --- |
| `sav_train` | train | `{video}.mp4`와 `{video}_manual.json`, 마스크는 RLE |
| `sav_val`, `sav_test` | vos | `JPEGImages_24fps/{video}/{frame}.jpg`와 `Annotations_6fps/{video}/{object}/{frame}.png` |

전체가 아니라 subset만 추출합니다. 기본값은 서로 겹치지 않는 비디오 구간에서 인코더 32개와 디코더 60개 샘플을 생성합니다. 전체 `sav_val.tar`는 155개 비디오에 15 GB이지만, 스크립트는 120개 비디오와 비디오당 8개의 어노테이션 프레임만 남기고 어노테이션이 없는 프레임은 버립니다. `--dry-run`으로 선택 결과와 크기를 미리 확인할 수 있습니다.

프레임 단위로 해당 프레임의 모든 객체 마스크를 함께 보존하므로, 디코더 캘리브레이션이 프레임당 임의의 객체 하나만 받는 대신 객체 크기 균형을 그대로 맞출 수 있습니다.

스크립트는 실제 확보된 비디오 수에 맞춘 skip 값과 함께 다음 명령을 그대로 출력합니다:

```bash
python prepare_calibration.py --stage both --defer-manifest \
  --sav-root ./data/sav --seed 1234 \
  --encoder-samples 32 --encoder-skip-videos 0 --encoder-max-videos 32 \
  --decoder-samples 60 --decoder-skip-videos 36 --decoder-max-videos 60
```

### 하나의 split으로 캘리브레이션과 평가를 모두 수행

하나의 split만으로 둘 다 수행할 수 있습니다. skip 오프셋이 해당 split에서 **서로 겹치지 않는 비디오 구간**을 잘라내기 때문입니다. 인코더 캘리브레이션, 디코더 캘리브레이션, 평가 사이에 공유되는 비디오가 없습니다. 이는 원래 레시피가 `sav_train` 안에서 사용한 것과 동일한 구성이며, 규모만 작을 뿐입니다.

`prepare_sav.py`는 실제 확보된 비디오 수에 맞춘 세 구간을 출력합니다. `sav_val.tar`의 155개 비디오 기준:

```text
disjoint video ranges (no video is shared between them):
  encoder calibration :   0 -  31
  decoder calibration :  36 -  95
  evaluation reserve  : 100 - 154  (55 videos)
```

출력되는 명령에는 `--encoder-max-videos`와 `--decoder-max-videos`가 포함되며, 이 두 옵션이 각 구간을 **강제 상한**으로 만듭니다. 장식이 아닙니다. 구간 크기를 `samples / per_video`로 계산할 수 없기 때문입니다. 하나의 비디오가 요청한 만큼 샘플을 내주지 못할 수 있습니다(`iter_frame_samples`는 지터가 적용된 프레임 인덱스를 집합으로 만들어 하나로 합쳐질 수 있고, `build_prompt()`는 포인트를 놓기에 너무 얇은 마스크를 거부합니다). 상한이 없으면 인코더 세트가 자기 구간을 넘어 디코더 구간으로 들어가고, 디코더 세트는 평가용으로 남겨 둔 구간까지 잠식합니다. 아무 예외도 발생하지 않으므로 조용히 진행됩니다. 그래서 구간은 비디오당 샘플 1개라는 최악의 경우를 기준으로 잡습니다.

상한이 있으면 공간이 부족할 때 조용히 넘어가지 않고 명시적으로 실패합니다(`requested 32 encoder samples, wrote 4`). 순서는 `--seed`로 섞이므로 `prepare_calibration.py`에 같은 `--seed`를 전달하면 구간이 그대로 재현됩니다.

중요한 것은 한 비디오가 두 구간에 나타나지 않는다는 점이며, 어떤 split에서 가져오는지는 문제가 되지 않습니다. 다운로드가 가능하다면 `sav_train`이 본래 의도된 소스이지만, `sav_val`이나 `sav_test`도 동일하게 동작합니다.

### 비디오 개수

기본값은 제공된 `sav_val` 아카이브를 대상으로 합니다:

| 세트 | 계산 | 필요 비디오 수 |
| --- | --- | ---: |
| 인코더 | skip 0 + 샘플 32 / 비디오당 2 | 32 |
| 디코더 | skip 36 + 샘플 60 / 비디오당 4 | 60 |

처음 96개 위치가 두 캘리브레이션 구간과 그 사이의 4개 비디오 간격을 포함합니다. 기본 120개 비디오 추출은 평가용으로 20개 비디오도 남깁니다. hard cap이 있어 샘플이 부족하면 평가 구간을 조용히 소비하지 않고 실패합니다.

> **참고**: 비디오 선택은 `--seed`로 섞이므로, 나중에 비디오를 더 추출하면 분할이 다시 섞이고 이전에 생성한 캘리브레이션이 더 이상 같은 비디오에 대응하지 않습니다. 준비를 끝낸 뒤 생성하거나, 두 세트를 함께 다시 생성하십시오.

## 단계 1: MBLT 그래프 생성

두 부분 모두 qbcompiler SAM2 참조 구현의 direct torch-parser 경로를 사용합니다.

```bash
python sam2_encoder_to_mblt.py         # 인코더: SAM2 -> MBLT
python sam2_decoder_to_mblt.py        # 디코더: SAM2 -> MBLT
```

두 명령은 다른 작업 디렉터리에서 절대 경로로 실행해도 기본적으로 이
`compilation/mask_generation` 디렉터리에 결과를 작성합니다.
다른 위치를 사용하려면 각 명령에 `--save-path /path/to/output/<part>.mblt`를 전달하십시오.

**인코더**는 `forward_image` 캡처를 legacy parser로 직접 파싱하여 세 FPN feature map을 노출합니다. **디코더**는 `sam_mask_decoder`를 직접 파싱하며 host bridge에서 출력 토큰 concat과 `image_embeddings + dense` 합성을 수행합니다. 두 명령은 `sam2_hiera_large_encoder.mblt`와 `sam2_hiera_large_decoder.mblt`를 생성합니다. `sparse_prompt_embeddings`의 프롬프트 축은 동적으로 표시되며, 이 튜토리얼은 1~3 포인트를 지원합니다. 그래프만 확인하려면 `--ignore-weight`를 쓰고, CUDA 메모리가 부족하면 `--torch-device cpu`를 사용하십시오.
### Direct-export 검증

두 exporter는 파싱 전에 실제 SAM2 호출을 캡처합니다. weight 직렬화 없이 그래프 구성을 확인하려면 `--ignore-weight`를 사용하고, CUDA 메모리가 부족하면 `--torch-device cpu`를 사용하십시오.

### 동적 프롬프트 축

추적된 프롬프트 길이에 고정되지 않도록 `sam2_decoder_to_mblt.py`가 `sparse_prompt_embeddings`의 축을 동적으로 표시합니다. 따라서 1~3 포인트 상호작용 프롬프트를 처리합니다.

### legacy parser를 사용하는 이유

인코더와 디코더는 qbcompiler SAM2 참조 구현과 같은 direct legacy-parser 경로를 사용하므로 두 MBLT가 일관된 파싱 및 직렬화 경로를 공유합니다.


## 단계 1-1: 디코더 input name 확인

디코더 input name은 파싱된 그래프에 따라 달라지며, 단계 2와 단계 3이 모두 이 이름에 의존합니다. MBLT가 실제로 보고하는 이름을 출력합니다:

```bash
python -c "from decoder_bindings import read_mblt_input_names; print(read_mblt_input_names('./sam2_hiera_large_decoder.mblt'))"
```

결과를 `decoder_input_bindings.json`과 비교하십시오. 이름이 다르면 해당 이름을 동일한 6개 semantic role에 매핑하는 새 파일을 작성하고 단계 2와 단계 3에서 `--decoder-input-bindings`로 전달합니다. 기본값에 맞추려고 캘리브레이션 텐서 순서를 바꾸지 마십시오. 여러 디코더 입력이 같은 shape을 가지므로 위치로 추측하면 오류 없이 조용히 잘못된 결과가 나옵니다.

## 단계 2: 캘리브레이션 데이터셋 준비

`prepare_calibration.py`는 SA-V manual masklet에서 두 캘리브레이션 세트를 생성합니다. 인코더 캘리브레이션에는 프레임만 필요하지만, 디코더 캘리브레이션에는 객체 내부에 포인트 프롬프트를 배치하기 위한 ground-truth 마스크도 필요합니다.

단계 0 이후 두 세트를 한 번에 생성합니다. 이제 `--sav-root`와 분리된 비디오 구간 인자는 기본값이므로, 디코더 MBLT만 있으면 됩니다:

```bash
python prepare_calibration.py
```

출력은 현재 작업 디렉터리가 아니라 스크립트와 같은 위치에 저장되므로, 어디에서 실행하든 튜토리얼의 `calib/` 트리가 동일한 자리에 생성됩니다:

- `calib/encoder/encoder_calib.txt`: 인코더 텐서 경로 목록
- `calib/encoder/encoder/*.npy`: 인코더 텐서
- `calib/decoder/decoder_calib.json`: input name, slot role, 텐서 경로가 포함된 디코더 manifest
- `calib/decoder/decoder/<role>/*.npy`: semantic role별 디렉터리에 저장된 디코더 텐서

`model_compile.py`도 기본값으로 동일한 두 경로를 읽습니다. 경로를 바꾸려면 `--encoder-output-dir` / `--decoder-output-dir`와 대응하는 `--encoder-calib` / `--decoder-calib`를 함께 지정하십시오.

인코더와 디코더 세트는 서로 겹치지 않는 비디오 구간(`--encoder-skip-videos 0`, `--decoder-skip-videos 36`)에서 추출되므로 두 모델이 동일한 영상으로 캘리브레이션되지 않습니다. `--stage encoder` 또는 `--stage decoder`로 한 세트씩 생성할 수도 있습니다.

### 인코더 캘리브레이션

인코더 샘플은 공식 SAM2 변환의 결과인 float32 NHWC `[1, 1024, 1024, 3]` 텐서입니다. 다른 shape이 나오면 스크립트가 즉시 실패하는데, 전처리가 조용히 바뀌면 인코더가 잘못된 입력 분포로 양자화되기 때문입니다.

### 디코더 캘리브레이션과 동적 프롬프트 축

프롬프트 인코더는 포인트 하나당 임베딩 1개에 패딩 1개를 더해 내보내므로, 포인트가 `N`개인 프롬프트는 `N + 1`개 항목을 만듭니다. 기본값 `--point-mix 1,2,3`은 1/2/3 포인트 프롬프트를 순환하며 프롬프트 길이 2, 3, 4를 생성합니다. SAM2가 앞에 붙이는 출력 토큰 6개는 이제 디코더 그래프 안에서 concat되므로 호스트가 넘기는 텐서에는 나타나지 않습니다.

point mix에 서로 다른 값이 둘 이상 포함되면 manifest는 프롬프트 축을 동적으로 표시합니다:

```python
if len(set(points_per_sample)) > 1:
    shapes_by_role["sparse_prompt_embeddings"][2] = -1
```

이 처리가 없으면 컴파일된 디코더는 단일 토큰 길이에 고정되어 포인트 개수가 다른 프롬프트를 거부합니다. 하나의 프롬프트 크기만 지원할 의도가 아니라면 기본 mix를 유지하십시오.

### 디코더 manifest 미루기

디코더 manifest는 quantizer가 보는 input name을 기준으로 하며, 이는 `sam2_decoder_to_mblt.py`가 만든 디코더 `.mblt`에서 읽는 **post-parse** 이름입니다. 디코더 ONNX는 존재하지 않습니다. 디코더는 그 경로를 거치지 않습니다.

디코더 텐서 생성에는 모델이 전혀 필요하지 않습니다. 텐서는 공식 FP32 호스트 경로에서 생성되고 model input name이 아니라 semantic role로 저장되기 때문입니다. 모델이 필요한 것은 input name을 기록하는 manifest뿐입니다. `--defer-manifest`가 이 둘을 분리하며, 파싱 가능한 디코더보다 텐서가 먼저 준비된 경우에 유용합니다:

```bash
python prepare_calibration.py --stage decoder --defer-manifest
python prepare_calibration.py --stage manifest
```

이 명령은 role별 텐서와 `calib/decoder/decoder_tensor_meta.json`을 저장한 뒤 manifest를 생성하며, 그 사이에 SA-V나 FP32 인코더를 다시 실행하지 않습니다. 인코더 캘리브레이션은 모델 파일이 전혀 필요 없습니다.

인코더 캘리브레이션은 모델 파일이 필요 없으므로 두 경로에서 동일합니다.

## 단계 3: 모델 컴파일

`model_compile.py`는 인코더와 디코더에 대해 `mxq_compile`을 각각 한 번씩 호출합니다. 컴파일 전에 디코더 MBLT를 다시 읽어, 캘리브레이션 manifest가 다른 그래프로 생성되었다면 진행을 거부합니다:

```python
if info.get("input names") != model_inputs:
    raise ValueError(
        "decoder calibration input names do not match the MBLT. Regenerate calibration "
        "with this exact decoder MBLT instead of relying on positional same-shape inputs."
    )
```

같은 shape을 가진 입력들 사이에서 위치가 어긋나면 잘못된 텐서가 양자화되면서도 실행은 되는 모델이 생성되기 때문에 이 검증이 필요합니다.

컴파일 설정은 인자로 직접 전달하지 않고 `compile_config.json`에서 읽습니다:

```json
{
  "quantization": { "calibration": { "output": 1, "mode": 0 } },
  "resourceManagement": {
    "useGPUOnlyForCalibration": true,
    "weightMemory": { "method": 3 }
  },
  "llm": { "apply": false }
}
```

`weightMemory.method`와 `useGPUOnlyForCalibration`은 `mxq_compile` 키워드에 대응이 없는 `CompileConfig` 필드이므로 `CompileConfig.from_file`로 로드합니다.

다른 튜토리얼과 마찬가지로 `model_compile.py`는 `torch.cuda.is_available()`이 참이면 CUDA를 사용하고 그렇지 않으면 CPU로 대체하며, 컴파일 시작 전에 선택된 호스트 디바이스를 출력합니다.

컴파일을 실행합니다:

```bash
python model_compile.py --target-device aries-rb
```

`--part`로 컴파일할 모델을 선택합니다. 기본값은 `both`이며, 하나만 지정하면 그 모델만 컴파일합니다. 디코더를 건너뛰면 디코더 manifest 검증도 함께 생략됩니다:

```bash
python model_compile.py --part encoder
```

`--dry-run`을 추가하면 파일, manifest, MBLT 입력 계약만 검증하고 컴파일은 수행하지 않습니다.

출력:

- `sam2_hiera_large_encoder.mxq`
- `sam2_hiera_large_decoder.mxq`

인코더 MXQ는 848 MB MBLT에서 약 268 MB로 줄어듭니다.

[../../runtime/python/mask_generation/README.KR.md](../../runtime/python/mask_generation/README.KR.md)의 런타임 튜토리얼은 두 파일이 정확히 이 경로에 있다고 가정합니다.

### 디코더는 CPU에서 컴파일됩니다

디코더 MBLT는 서브그래프가 2개이므로 `mxq_compile`에 `cpu_offload=True`가 필요합니다. 기본값 `cpu_offload=False`로는 아예 거부됩니다:

```text
ValueError: cpu_offload=False cannot compile a model with 2 subgraphs
```

또한 host bridge가 CPU에 있는 상수를 캘리브레이션 텐서와 concat하므로 GPU에서는 양자화 도중 실패합니다:

```text
RuntimeError: Error. quantizeFB failed. Expected all tensors to be on the same device,
but found at least two devices, cpu and cuda:0!
```

그래서 `model_compile.py`는 디코더를 `cpu_offload=True`, `device="cpu"`로 컴파일하고, 인코더는 더 빠른 GPU 경로를 유지합니다. 자동으로 처리되므로 별도 플래그는 필요 없습니다.

### 대상 디바이스 선택 (`--target-device`)

| 사용자 | `--target-device` | 모델 |
| --- | --- | --- |
| ARIES | `aries-rb` | `sam2-hiera-large` |

> **참고**: SAM2는 `aries-rb`에서만 검증되었습니다. REGULUS는 이 튜토리얼의 범위에 포함되지 않습니다.

## 파라미터

`prepare_calibration.py`:

- `--stage`: 생성할 세트. `encoder`, `decoder`, `both` 중 선택. 기본값: `both`.
- `--sav-root`: 추출한 SA-V val/test 또는 train 루트. 기본값은 이 스크립트 옆 `data/sav`입니다.
- `--sam2-root`: `facebookresearch/sam2` 로컬 checkout.
- `--model-id`: SAM2 모델 id. 기본값: `facebook/sam2-hiera-large`.
- `--encoder-samples`: 인코더 샘플 개수. 기본값: `32`.
- `--decoder-samples`: 디코더 샘플 개수. 기본값: `60`.
- `--point-mix`: 디코더 샘플에 순환 적용할 포인트 개수. 기본값: `1,2,3`.
- `--encoder-skip-videos`, `--decoder-skip-videos`: 두 세트를 분리하기 위해 건너뛸 비디오 수. 기본값: `0`, `36`.
- `--decoder-model`: manifest가 post-parse input name과 일치해야 하는, `sam2_decoder_to_mblt.py`가 만든 디코더 `.mblt`.
- `--defer-manifest`: manifest를 생성하지 않고 디코더 텐서만 생성합니다.
- `--stage manifest`: 이미 저장된 텐서와 `--decoder-model`로부터 manifest를 생성합니다.
- `--decoder-input-bindings`: MBLT input name과 semantic role 매핑. 기본값: `./decoder_input_bindings.json`.

`model_compile.py`:

- `--encoder-mblt`, `--decoder-mblt`: 입력 MBLT 그래프.
- `--encoder-calib`, `--decoder-calib`: 단계 2에서 생성한 캘리브레이션 목록과 manifest.
- `--encoder-save-path`, `--decoder-save-path`: MXQ 출력 경로.
- `--compile-config`: `CompileConfig` JSON. 기본값: `./compile_config.json`.
- `--target-device`: 대상 NPU. 기본값: `aries-rb`.
- `--gpu`: 캘리브레이션에 사용할 CUDA 디바이스 인덱스. 기본값: `0`.
- `--part`: 컴파일할 모델. `encoder`, `decoder`, `both`. 기본값: `both`.
- `--dry-run`: 컴파일 없이 입력만 검증.

## 검증된 결과

다음 수치는 현재 direct-parser 워크플로우의 검증이 아니라, 이전 host-token-assembly 디코더 경로에서 측정한 historical 결과입니다:

| 경로 | 샘플 수 | mIoU |
| --- | ---: | ---: |
| Official FP32 | 200 | 0.775005 |
| 인코더 + 디코더 MXQ | 200 | 0.775706 |

FP32 대비 binary mask agreement는 `0.983084`, low-resolution logit cosine 유사도는 `0.998363`이었고, 이전 디코더는 토큰 길이 8, 9, 10을 처리했습니다.

이 수치를 재현하려면 별도의 평가 도구가 필요하며, 해당 내용은 이 튜토리얼의 범위를 벗어납니다.

이 수치는 출력이 4개였던 이전 Mobilint 모델 빌드의 디코더 MBLT로 측정한 값입니다. `qbcompiler` 빌드에 따라 단계 1에서 파싱한 디코더는 출력이 2개(mask, IoU)이거나 4개(SAM 토큰과 object score 추가)일 수 있습니다. 런타임 튜토리얼은 두 경우를 모두 처리하며, 분할 결과에 영향을 주는 것은 mask와 IoU뿐입니다.

## 이 튜토리얼의 파일

- `prepare_sav.py`: 다운로드한 SA-V 아카이브에서 캘리브레이션용 subset을 추출합니다
- `sam2_encoder_to_mblt.py`: 인코더를 legacy parser로 직접 MBLT에 내보냅니다
- `sam2_decoder_to_mblt.py`: legacy 파서로 마스크 디코더를 MBLT로 파싱합니다
- `prepare_calibration.py`: SA-V에서 인코더와 디코더 캘리브레이션 데이터를 생성합니다
- `model_compile.py`: 선택한 `--target-device`에 대해 두 MBLT 그래프를 MXQ로 컴파일합니다
- `sam2_host.py`: 파싱과 캘리브레이션 생성이 공유하는 호스트 측 SAM2 헬퍼
- `sav_dataset.py`: SA-V 프레임, 마스크, 포인트 프롬프트 샘플링
- `decoder_bindings.py`: MBLT input name과 semantic role 해석
- `decoder_input_bindings.json`: 기본 바인딩 매핑
- `compile_config.json`: 두 모델에 사용하는 `qbcompiler` `CompileConfig`
- `requirements.txt`: 캘리브레이션 생성을 위한 Python 의존성
- `README.KR.md`: 이 예제의 전체 워크플로우를 설명합니다
