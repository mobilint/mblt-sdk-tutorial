# BERT 컴파일

이 튜토리얼은 [Sentence-BERT](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors)를 Mobilint NPU용 MXQ 모델로 컴파일합니다.

입력 임베딩과 mean pooling은 CPU에서 실행하고 BERT encoder는 NPU에서 실행합니다.
임베딩 가중치에 회전이나 별도 변환을 적용하지 않습니다.
캘리브레이션과 컴파일은 Hugging Face 원본 모델을 직접 사용합니다.
마지막 준비 단계에서 원본 모델 파일과 컴파일된 MXQ를 하나의 런타임 디렉터리에 모읍니다.

모든 명령은 이 디렉토리에서 실행합니다.

## 사전 준비

```bash
pip install -r requirements.txt
```

## 1. 캘리브레이션 데이터 생성

```bash
python generate_calib.py \
  --model-id sentence-transformers-testing/stsb-bert-tiny-safetensors \
  --output-dir ./calibration_data
```

STS Benchmark 검증 세트에서 256개 문장을 선택하고, 원본 BERT 임베딩 레이어로 변환해 `./calibration_data`에 저장합니다.
출력 디렉토리는 비어 있어야 합니다.

## 2. 모델 컴파일

```bash
python compile_model.py \
  --model-id sentence-transformers-testing/stsb-bert-tiny-safetensors \
  --calib-data-path ./calibration_data \
  --mblt-path ./mblt/stsb-bert-tiny-safetensors.mblt \
  --save-path ./mxq/stsb-bert-tiny-safetensors.mxq \
  --target-device aries-rb
```

REGULUS용으로 컴파일하려면 다음과 같이 실행합니다.

```bash
python compile_model.py --target-device regulus-rb
```

MBLT 출력은 `./mblt/stsb-bert-tiny-safetensors.mblt`에 저장됩니다.
MXQ 출력은 `./mxq/stsb-bert-tiny-safetensors.mxq`에 저장됩니다.

### 지원 디바이스

| 디바이스 | 지원 여부 |
| --- | --- |
| `aries-rb` | 지원 |
| `regulus-rb` | 지원 |
| `regulus-ra` | 미지원 |

## 3. 런타임 모델 준비

MXQ 컴파일이 끝난 뒤 실행합니다.
`prepare_model.py`는 원본 모델 파일을 다운로드하고 컴파일된 MXQ를 하나의 런타임 디렉터리에 복사합니다.

```bash
python prepare_model.py \
  --model-id sentence-transformers-testing/stsb-bert-tiny-safetensors \
  --mxq-path ./mxq/stsb-bert-tiny-safetensors.mxq \
  --output-dir ./bert-mxq
```

`./bert-mxq`에는 런타임 예제에 필요한 모델 가중치, 설정, 토크나이저, pooling 설정과 컴파일된 MXQ가 모두 저장됩니다.

## 파일 구조

```text
bert/
├── compile_model.py
├── generate_calib.py
├── prepare_model.py
├── requirements.txt
├── README.md
├── README.KR.md
├── calibration_data/
│   └── *.npy
├── mblt/
│   └── stsb-bert-tiny-safetensors.mblt
├── mxq/
│   └── stsb-bert-tiny-safetensors.mxq
└── bert-mxq/
    ├── model.safetensors
    ├── stsb-bert-tiny-safetensors.mxq
    ├── 1_Pooling/config.json
    └── 토크나이저 및 모델 설정 파일
```

## 런타임

컴파일이 끝나면 [BERT Python 런타임](../../runtime/python/bert/README.KR.md)을 실행합니다.
