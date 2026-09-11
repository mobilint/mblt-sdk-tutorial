# 마스크 생성 모델 컴파일

이 튜토리얼은 Meta의 [SAM2 Hiera large](https://github.com/facebookresearch/sam2) 이미지 인코더와 마스크 디코더를 각각 MXQ로 컴파일합니다. 이미지 전처리, 프롬프트 인코더, 디코더 입력 준비, 마스크 업스케일링은 호스트에서 실행됩니다.

인코더와 디코더는 ONNX를 거치지 않습니다. 각 `compile_*.py`가 `mblt_compile()`로 MBLT를 만든 뒤 같은 스크립트에서 `mxq_compile()`을 실행합니다.

## 지원 디바이스

| 디바이스 | 지원 여부 |
| --- | --- |
| `aries-rb` | 지원 |
| `regulus-rb` | 지원 |
| `regulus-ra` | 미지원 |

기본값은 `aries-rb`입니다.

## 사전 준비

- Python 3.10 이상
- `transformers==5.16.1`
- [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- 사용자가 직접 다운로드한 SA-V 아카이브

필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

SAM2는 공식 저장소에서 설치합니다.

```bash
git clone https://github.com/facebookresearch/sam2.git /workspace/sam2
pip install -e /workspace/sam2
```

## 1. SA-V 데이터 준비

SA-V는 공식 [데이터셋 안내](https://github.com/facebookresearch/sam2/blob/main/sav_dataset/README.md)에 따라 직접 다운로드합니다. `prepare_sav.py`는 다운로드 기능이 없으며 이미 받은 tar에서 필요한 subset만 추출합니다.

기본 파일명은 `sav_val.tar`입니다.

```bash
python prepare_sav.py
```

다른 파일명을 사용하면 경로를 지정합니다.

```bash
python prepare_sav.py --archive /path/to/sav_000.tar
```

기본 설정은 120개 비디오를 `./data/sav`에 준비합니다. 인코더, 디코더, 평가용 비디오 구간은 서로 겹치지 않습니다.

| 용도 | 비디오 위치 |
| --- | --- |
| 인코더 캘리브레이션 | 0–31 |
| 디코더 캘리브레이션 | 36–95 |
| 평가용 예비 데이터 | 100 이후 |

같은 `--seed`를 사용하면 비디오와 샘플 선택이 재현됩니다.

## 2. 캘리브레이션 데이터 생성

인코더와 디코더 캘리브레이션 텐서를 함께 생성합니다.

```bash
python prepare_calibration.py
```

한쪽만 생성할 수도 있습니다.

```bash
python prepare_calibration.py --stage encoder
python prepare_calibration.py --stage decoder
```

주요 출력은 다음과 같습니다.

```text
calib/encoder/encoder_calib.txt
calib/encoder/encoder/*.npy
calib/decoder/decoder_tensor_meta.json
calib/decoder/decoder/<role>/*.npy
```

인코더 입력은 공식 SAM2 전처리를 적용한 float32 NHWC `[1, 1024, 1024, 3]`입니다. 디코더는 기본적으로 1개, 2개, 3개 포인트 프롬프트를 순환하여 생성합니다.

## 3. 인코더 컴파일

ARIES용 인코더를 컴파일합니다.

```bash
python compile_encoder.py --target-device aries-rb
```

REGULUS용 인코더를 컴파일하려면 다음과 같이 실행합니다.

```bash
python compile_encoder.py --target-device regulus-rb
```

스크립트는 다음 순서로 실행됩니다.

1. 실제 `predictor.set_image()` 입력을 캡처합니다.
2. `mblt/<target-device>/sam2_hiera_large_encoder.mblt`를 생성합니다.
3. 인코더 캘리브레이션 데이터로 `mxq/<target-device>/sam2_hiera_large_encoder.mxq`를 생성합니다.

## 4. 디코더 컴파일

ARIES용 디코더를 컴파일합니다.

```bash
python compile_decoder.py --target-device aries-rb
```

REGULUS용 디코더를 컴파일하려면 다음과 같이 실행합니다.

```bash
python compile_decoder.py --target-device regulus-rb
```

스크립트는 다음 순서로 실행됩니다.

1. 실제 `predictor.predict()` 입력을 캡처합니다.
2. `mblt/<target-device>/sam2_hiera_large_decoder.mblt`를 생성합니다.
3. 생성된 MBLT의 입력명을 읽어 `calib/decoder/decoder_calib.json`을 생성합니다.
4. 디코더 캘리브레이션 데이터로 `mxq/<target-device>/sam2_hiera_large_decoder.mxq`를 생성합니다.

디코더의 `tokens` 축은 동적입니다. 따라서 캘리브레이션에 포함된 1~3개 포인트 프롬프트를 처리할 수 있습니다.

## 디코더 입력 계약

디코더에는 입력이 6개 있으며, 그중 세 입력은 shape이 같습니다. 따라서 위치를 추측하지 않고 MBLT 입력명을 semantic role에 연결합니다.

```text
tokens/reshape                  -> tokens        (1, 1,    T, 256)
add/transpose                   -> src_plus_pos  (1, 1, 4096, 256)
flatten/reshape/transpose       -> src           (1, 1, 4096, 256)
flatten_1/reshape/transpose     -> pos_src       (1, 1, 4096, 256)
high_res_features_1/transpose  -> hrf1_nhwc     (1, 128, 128, 64)
high_res_features_0/transpose  -> hrf0_nhwc     (1, 256, 256, 32)
```

이 매핑은 `decoder_input_bindings.json`에 있습니다. `compile_decoder.py`가 방금 생성한 MBLT에서 입력명을 읽어 manifest를 만들기 때문에 오래된 manifest가 다른 그래프에 사용되지 않습니다.

## 출력

`aries-rb`를 선택한 경우 다음 파일이 생성됩니다.

```text
mblt/aries-rb/sam2_hiera_large_encoder.mblt
mblt/aries-rb/sam2_hiera_large_decoder.mblt
mxq/aries-rb/sam2_hiera_large_encoder.mxq
mxq/aries-rb/sam2_hiera_large_decoder.mxq
```

`regulus-rb`를 선택하면 같은 파일이 `mblt/regulus-rb`와 `mxq/regulus-rb`에 생성됩니다.

## 주요 옵션

`compile_encoder.py`와 `compile_decoder.py`:

- `--target-device`: `aries-rb` 또는 `regulus-rb`. 기본값은 `aries-rb`입니다.
- `--model-id`: Hugging Face SAM2 모델 ID.
- `--image`: MBLT 입력 캡처에 사용할 이미지.
- `--device`: 호스트 SAM2 모델을 실행할 torch 디바이스. 기본값은 `cuda`이며 CUDA를 사용할 수 없으면 CPU를 사용합니다.

`prepare_calibration.py`:

- `--stage`: `encoder`, `decoder`, `both`. 기본값은 `both`입니다.
- `--sav-root`: 추출한 SA-V 루트. 기본값은 `./data/sav`입니다.
- `--encoder-samples`: 인코더 샘플 수. 기본값은 32입니다.
- `--decoder-samples`: 디코더 샘플 수. 기본값은 60입니다.
- `--point-mix`: 디코더 포인트 개수. 기본값은 `1,2,3`입니다.
- `--seed`: 샘플 선택 seed. 기본값은 1234입니다.

## 파일 구성

- `prepare_sav.py`: SA-V tar에서 캘리브레이션 subset을 추출합니다.
- `prepare_calibration.py`: 인코더 및 디코더 캘리브레이션 텐서를 생성합니다.
- `compile_encoder.py`: 인코더 MBLT와 MXQ를 생성합니다.
- `compile_decoder.py`: 디코더 MBLT, 캘리브레이션 manifest, MXQ를 생성합니다.
- `sam2_host.py`: 캘리브레이션과 컴파일이 공유하는 SAM2 호스트 처리를 제공합니다.
- `decoder_bindings.py`: 디코더 MBLT 입력명과 semantic role을 연결합니다.
- `compile_config.json`: MXQ 컴파일 설정입니다.

추론 방법은 [런타임 튜토리얼](../../runtime/python/mask_generation/README.KR.md)을 참고하십시오.
