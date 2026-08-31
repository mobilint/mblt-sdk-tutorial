# 마스크 생성 런타임

이 튜토리얼은 컴파일된 `SAM2 Hiera large` MXQ 모델을 Mobilint `qbruntime`으로 실행하는 방법을 설명합니다.

시작하기 전에 [../../../compilation/mask_generation/README.KR.md](../../../compilation/mask_generation/README.KR.md)의 컴파일 과정을 먼저 완료하세요. 이 디렉토리의 런타임 예제는 컴파일된 모델이 `../../../compilation/mask_generation/sam2_hiera_large_encoder.mxq`와 `../../../compilation/mask_generation/sam2_hiera_large_decoder.mxq`에 있다고 가정합니다.

## 사전 준비

다음 항목이 준비되어 있어야 합니다:

- Mobilint `qbruntime`
- 컴파일된 `.mxq` 모델 파일 두 개
- [facebookresearch/sam2](https://github.com/facebookresearch/sam2) 로컬 checkout
- Python 패키지: `torch`, `numpy`, `pillow`

Python 패키지가 설치되어 있지 않다면 다음과 같이 설치합니다:

```bash
pip install -r requirements.txt
```

SAM2는 PyPI에 없으므로 공식 저장소에서 설치합니다:

```bash
git clone https://github.com/facebookresearch/sam2.git /workspace/sam2
pip install -e /workspace/sam2
```

패키지 자체를 설치하지 않으려면 원하는 위치에 clone한 뒤 `--sam2-root`로 경로를 전달하십시오. 이 옵션은 checkout을 `sys.path`에 추가할 뿐이므로 의존성은 따로 설치해야 합니다(`pip install -r /path/to/sam2/requirements.txt` 또는 `pip install -e`). 그렇지 않으면 `sam2.sam2_image_predictor` import가 실패합니다.

SAM2 체크포인트는 최초 실행 시 Hugging Face에서 다운로드되므로, 런타임 호스트에는 네트워크 접근 또는 미리 준비된 Hugging Face 캐시가 필요합니다.

## 개요

SAM2는 프롬프트 기반 모델입니다. 이미지 인코더는 이미지당 한 번, 마스크 디코더는 프롬프트당 한 번 실행됩니다. 컴파일 대상은 이 두 단계뿐이며, 나머지는 호스트에서 공식 SAM2 코드로 실행됩니다.

런타임 흐름은 `inference_mxq.py`에 구현되어 있으며 다음 순서를 따릅니다:

1. 두 MXQ 모델을 `qbruntime`으로 로드하고 각각 별도의 코어에 배치합니다.
2. 공식 SAM2 이미지 변환을 적용해 `[1024, 1024, 3]` float32 입력을 만듭니다.
3. 인코더 MXQ를 Mobilint NPU에서 실행해 FPN 특징 3개를 얻습니다.
4. 해당 특징을 호스트 predictor에 설치하고 프롬프트 인코더를 실행합니다.
5. 조립된 디코더 입력 6개를 디코더 MXQ에 전달합니다.
6. 마스크 logit을 원본 이미지 크기로 업스케일하고 오버레이를 렌더링합니다.

```text
image
  -> SAM2 이미지 변환                          호스트
  -> 이미지 인코더                             인코더 MXQ
  -> 프롬프트 인코더 및 토큰 조립               호스트
  -> 마스크 디코더 본체                        디코더 MXQ
  -> 마스크 업스케일링                         호스트
```

## 이 튜토리얼의 파일

- `inference_mxq.py`: 두 모델 파이프라인 전체를 실행하고 오버레이와 원시 출력을 저장합니다.
- `sam2_host.py`: 전처리, 특징 설치, 프롬프트 인코딩, 마스크 업스케일링을 위한 호스트 측 SAM2 헬퍼입니다.
- `contracts.py`: 디코더 입력 순서, shape 검증, 디코더 출력 식별을 담당합니다.
- `visualize.py`: 각 마스크 후보에 프롬프트 포인트를 함께 렌더링합니다.
- `requirements.txt`: 호스트 SAM2 코드의 Python 의존성입니다.

## 스크립트 동작 방식

### 두 모델 실행

두 모델이 동시에 상주하므로, 같은 코어를 함께 점유하지 않도록 각각 별도의 코어에 배치합니다:

```python
def launch_model(path: str, accelerator: qbruntime.Accelerator, core: qbruntime.Core) -> qbruntime.Model:
    model_config = qbruntime.ModelConfig()
    model_config.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, core)])
    model = qbruntime.Model(path, model_config)
    model.launch(accelerator)
    return model


accelerator = qbruntime.Accelerator()
encoder = launch_model(args.encoder_mxq, accelerator, qbruntime.Core.Core0)
decoder = launch_model(args.decoder_mxq, accelerator, qbruntime.Core.Core1)
```

`Accelerator`는 한 번만 생성해 두 모델이 dispose될 때까지 유지하며, 각 `launch` 호출에 임시 객체로 전달하지 않습니다.

### 배치 차원

SAM2 호스트 코드는 바깥쪽 배치 축이 있는 텐서를 만들지만, `qbruntime`은 버퍼 shape에서 그 축을 생략합니다. 따라서 모든 입력은 추론 전에 배치 축을 제거합니다:

```python
def strip_runtime_batch(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim >= 4 and value.shape[0] == 1:
        value = value[0]
    return np.ascontiguousarray(value, dtype=np.float32)
```

이는 `np.expand_dims`로 배치 차원을 추가한 뒤 `infer`를 호출하는 이 저장소의 단일 입력 비전 튜토리얼과 반대입니다. 해당 패턴을 여기에 그대로 옮기지 마십시오.

### 인코더 출력

인코더는 FPN 3단계를 반환합니다. 스크립트는 한 축이 아니라 **전체 shape**으로 각 단계를 식별하므로, 런타임이 NHWC를 보고하든 NCHW를 보고하든 동작합니다:

```python
FPN_LEVELS_CHW = ((32, 256, 256), (64, 128, 128), (256, 64, 64))
nhwc = {(h, w, c): c for c, h, w in FPN_LEVELS_CHW}
nchw = {(c, h, w): c for c, h, w in FPN_LEVELS_CHW}
...
if shape in nchw:
    channel = nchw[shape]
    tensor = torch.from_numpy(np.ascontiguousarray(array))[None]
elif shape in nhwc:
    channel = nhwc[shape]
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None]
```

전체 shape을 대조해야 모호함이 없습니다. 마지막 축만 검사하면 NCHW를 잘못 읽습니다. `(32, 256, 256)`은 `256`으로 끝나므로 256채널 NHWC로, `(256, 64, 64)`는 64채널로 오인됩니다.

이후 세 단계를 호스트 predictor에 설치하면, predictor는 자체 인코더를 건너뛰고 NPU 특징을 사용해 프롬프트 인코딩과 마스크 업스케일링을 수행합니다.

### 디코더 입력 순서

파이프라인에서 가장 실수하기 쉬운 부분입니다. 디코더에는 입력이 6개 있고 그중 3개가 `(1, 256, 64, 64)`로 같은 shape이므로 위치만으로는 구분할 수 없습니다.

여기서 사용하는 런타임 위치 순서:

```text
image_embeddings, dense_prompt_embeddings, image_pe, sparse_prompt_embeddings, hrf0_nhwc, hrf1_nhwc
```

이는 캘리브레이션과 컴파일에 사용하는 MBLT input name 순서와 동일하므로, 기억해야 할 순서는 하나뿐입니다. NPU 없이도 자신의 아티팩트에서 확인할 수 있습니다:

```bash
python -c "import qbruntime; print(qbruntime.get_model_summary('../../../compilation/mask_generation/sam2_hiera_large_decoder.mxq'))"
```

출력:

```text
Input - Shapes: [(256, 64, 64), (256, 64, 64), (256, 64, 64), (1, -1, 256),
                 (256, 256, 32), (128, 128, 64)]
```

`-1`이 프롬프트 축이므로 컴파일된 디코더가 하나의 프롬프트 크기에 고정되지 않습니다. 이 튜토리얼이 지원하는 범위는 1~3 포인트이며, `inference_mxq.py`는 그 범위를 벗어나면 추론 전에 거부합니다. 순서를 잘못 전달하면 오류가 아니라 그럴듯하지만 잘못된 마스크가 나오므로, `contracts.py`는 semantic role로 입력을 구성한 뒤 런타임 shape과 대조합니다:

```python
decoder_feed = build_decoder_runtime_feed(decoder_tensors, args.decoder_runtime_order)
validate_runtime_shapes(decoder_feed, decoder.get_model_input_shape(), "decoder")
```

다른 디코더 MBLT로 다시 컴파일한 경우 `get_model_summary`로 semantic 순서를 복원하려 하지 **마십시오**. shape만 출력되는데 앞의 세 입력이 모두 `(256, 64, 64)`이므로, 그 셋 사이에서 추측하면 `image_embeddings`, `dense_prompt_embeddings`, `image_pe`가 서로 뒤바뀐 채 모든 shape 검사를 통과하고 그럴듯하지만 잘못된 마스크가 나옵니다.

대신 해당 MBLT로 생성한 캘리브레이션 manifest의 `slot roles` 순서를 읽어 `--decoder-runtime-order`로 전달하십시오:

```bash
python -c "import json; print(json.load(open('../../../compilation/mask_generation/calib/decoder/decoder_calib.json'))['info']['slot roles'])"
```

### 디코더 출력

디코더는 `output_meta=lambda x: x[0][:2]`로 파싱되므로 출력은 mask와 IoU 2개입니다. 이전 wrapper 추적 디코더는 SAM 토큰과 object score도 내보냈으며, 존재하는 경우 그대로 처리합니다.

qbruntime은 런타임 출력 순서가 컴파일된 그래프의 선언 순서와 일치한다고 보장하지 않으므로, 출력은 위치가 아니라 원소 개수로 구분합니다. 마스크 출력은 `256 x 256`의 배수이고, IoU 점수는 마스크 개수와 같으며, SAM 토큰은 `num_masks x 256`이고, object score는 값 하나입니다. 모든 출력은 NaN과 무한대 여부도 검사합니다. 비유한 값이 그대로 통과하면 IoU `argmax`와 `> 0` 마스크 임계값을 조용히 지나가 실패를 보고하는 대신 예측을 오염시키기 때문입니다.

분할 결과에 영향을 주는 것은 mask와 IoU뿐입니다. 스크립트는 마스크 후보 3개를 모두 보고하며, 예측 IoU가 가장 높은 후보를 `selected`로 표시합니다.

## 예제 실행

SAM2는 프롬프트가 가리키는 대상을 분할하므로 프롬프트는 필수입니다. 포인트는 원본 이미지 좌표의 `X,Y,LABEL` 형식이며 `1`은 positive, `0`은 negative입니다:

```bash
python inference_mxq.py --point 500,580,1
```

이 명령은 다음 기본값을 사용합니다:

- 인코더 모델: `../../../compilation/mask_generation/sam2_hiera_large_encoder.mxq`
- 디코더 모델: `../../../compilation/mask_generation/sam2_hiera_large_decoder.mxq`
- 입력 이미지: `../rc/bus.jpg`
- 출력 디렉토리: `./tmp/demo`

경로를 명시적으로 전달하거나 positive와 negative 포인트를 함께 사용하려면 다음과 같이 실행합니다:

```bash
python inference_mxq.py --encoder-mxq ../../../compilation/mask_generation/sam2_hiera_large_encoder.mxq --decoder-mxq ../../../compilation/mask_generation/sam2_hiera_large_decoder.mxq --image-path ../rc/bus.jpg --output-dir ./tmp/custom --point 500,580,1 --point 400,120,0
```

디코더는 포인트 1~3개를 지원합니다. 캘리브레이션에서 여러 포인트 개수를 혼합해 토큰 축을 동적으로 표시했기 때문에 컴파일된 모델이 이 범위를 지원합니다.

## 파라미터

- `--encoder-mxq`: 컴파일된 인코더 MXQ 모델 경로.
- `--decoder-mxq`: 컴파일된 디코더 MXQ 모델 경로.
- `--image-path`: 입력 이미지 경로. 기본값: `../rc/bus.jpg`.
- `--point`: 원본 이미지 좌표의 `X,Y,LABEL` 프롬프트 포인트. 최대 3개까지 반복 지정. 필수.
- `--output-dir`: 오버레이와 원시 출력을 저장할 디렉토리. 기본값: `./tmp/demo`.
- `--sam2-root`: `facebookresearch/sam2` 로컬 checkout.
- `--model-id`: SAM2 모델 id. 기본값: `facebook/sam2-hiera-large`.
- `--torch-device`: 호스트 SAM2 코드가 사용할 torch 디바이스. 사용 가능하면 `cuda`, 그렇지 않으면 `cpu`가 기본값입니다.
- `--decoder-runtime-order`: semantic 입력 순서(쉼표 구분). 다시 빌드한 디코더라면 캘리브레이션 manifest의 `info['slot roles']`에서 읽으십시오. shape만 출력하는 런타임 요약으로는 `(256, 64, 64)` 입력 3개를 구분할 수 없습니다.

## 예상 출력

출력 디렉토리에는 다음이 생성됩니다:

- `mask_0.png`, `mask_1.png`, `mask_2.png`: 원본 이미지 위에 마스크 후보 3개와 프롬프트 포인트를 오버레이한 결과
- `outputs.npz`: binary mask, 원본 해상도 logit, low-resolution logit, 예측 IoU, 선택된 인덱스. 출력이 4개인 디코더에서는 SAM 토큰과 object score도 포함
- `summary.json`: 프롬프트, 후보별 예측 IoU, 선택된 후보 인덱스

## 참고 사항

- 이 튜토리얼은 `use_multimask_token_for_obj_ptr=True`인 SAM2 Hiera 디코더 계약을 대상으로 합니다. 다른 SAM2 디코더 variant는 지원하지 않습니다.
- box 프롬프트와 mask 프롬프트는 지원하지 않으며 포인트 프롬프트만 지원합니다.
- 프롬프트 인코더와 마스크 업스케일링이 호스트에서 실행되므로 호스트 SAM2 모델은 여전히 전체가 로드됩니다. 스크립트는 CUDA를 사용할 수 있으면 CUDA를 선택하고, 그렇지 않으면 CPU로 대체합니다. CPU는 느리지만 CPU 전용 런타임 호스트에서도 동작합니다.
- 코어 배치는 더 이상 권장되지 않는 `set_single_core_mode(num_cores=...)` 형식을 대체합니다. 컴파일 가이드에 인용된 정확도 수치는 이전 실행 구성에서 측정된 값입니다.
- 컴파일에 사용한 `.mblt` 파일은 [Mobilint Netron](https://netron.mobilint.com/)에서 입력과 출력 텐서를 확인할 수 있습니다.
- 전체 실행에는 정상적으로 구성된 Mobilint 런타임 환경과 호환 하드웨어가 필요합니다.
