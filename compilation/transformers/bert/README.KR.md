# Bidirectional Encoder Representations from Transformers

이 튜토리얼에서는 모빌린트 Qubee 컴파일러를 사용하여 BERT 모델을 컴파일하는 상세한 방법을 제공합니다. 컴파일 과정은 표준 BERT 모델을 모빌린트 NPU 하드웨어에서 효율적으로 실행될 수 있는 최적화된 `.mxq` 포맷으로 변환합니다.

이 튜토리얼에서는 [Korean Sentence Bert (STS finetuned)](https://huggingface.co/jhgan/ko-sbert-sts) 모델을 사용합니다.

## 전제 조건 (Prerequisites)

시작하기 전에 다음 항목들이 설치되어 있는지 확인해 주세요:

- Qubee SDK 컴파일러 (버전 0.12 이상 필수)
- CUDA를 지원하는 GPU (컴파일 시간 단축을 위해 권장)
- HuggingFace 계정 (Llama 모델 등 접근 제한 모델 사용 시 필요)

또한, 다음 패키지들을 설치해야 합니다:

```bash
pip install accelerate datasets
```

## 모델 분석 (Model Analysis)

BERT의 복잡한 구조로 인해 일부 레이어는 컴파일러에서 지원되지 않을 수 있습니다. 따라서 컴파일 과정에 앞서 모델 분석을 수행해야 합니다.

모델 구조를 분석하기 위해, 모델을 MBLT 포맷으로 변환합니다.

```python
from qubee import mblt_compile
from qubee.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qubee.model_dict.parser.backend.torch.util import wrap_tensor
from transformers import BertModel, BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(
        "jhgan/ko-sbert-sts",
        trust_remote_code=True,
    )
    model = BertModel.from_pretrained("jhgan/ko-sbert-sts", trust_remote_code=True)
    model.eval()

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    feed_dict = {}
    for k, v in inputs.items():
        wrapped = wrap_tensor(k, v)
        wrapped.src_shape[1].set_dynamic()
        feed_dict[k] = wrapped
    set_attention_mask(feed_dict["attention_mask"], "padding_mask")

    mblt_compile(
        model=model,
        mblt_save_path="ko-sbert-sts.mblt",
        backend="torch",
        feed_dict=feed_dict,
        cpu_offload=True,
    )

```

`mblt_compile.py` 코드를 실행하면 `ko-sbert-sts.mblt` 파일을 얻게 됩니다. [Netron](https://netron.mobilint.com)을 사용하여 모델의 아키텍처를 시각화해 볼 수 있습니다.

Netron 시각화 결과에 따르면, 입력 임베딩(embedding)과 관련된 일부 레이어가 지원되지 않는 것을 확인할 수 있습니다. 따라서 지원되는 레이어의 입력에 맞춰 캘리브레이션 데이터셋(calibration dataset)을 준비하고, 지원되지 않는 연산은 모빌린트 NPU 외부에서 처리해야 합니다.

## 캘리브레이션 데이터셋 준비 (Calibration Dataset Preparation)

### 임베딩 가중치 추출 (Get Embedding Weights)

Netron에서 확인했듯이 입력 임베딩 부분은 지원되지 않습니다. 따라서 지원되는 레이어의 입력에 맞추기 위해 캘리브레이션 데이터셋을 준비해야 합니다. 이를 위해 모델에서 임베딩 가중치를 추출하여 `.pth` 파일로 저장합니다.

이는 `get_embedding.py` 코드를 실행하여 수행할 수 있습니다.

```python
import torch
from transformers import BertModel

model = BertModel.from_pretrained("jhgan/ko-sbert-sts", trust_remote_code=True)

word_embeddings = model.embeddings.word_embeddings.weight
token_type_embeddings = model.embeddings.token_type_embeddings.weight
position_embeddings = model.embeddings.position_embeddings.weight
layernorm_weight = model.embeddings.LayerNorm.weight
layernorm_bias = model.embeddings.LayerNorm.bias

print(word_embeddings.shape)
print(token_type_embeddings.shape)
print(position_embeddings.shape)
print(layernorm_weight.shape)
print(layernorm_bias.shape)
weight_dict = {
    "word_embeddings": word_embeddings,
    "token_type_embeddings": token_type_embeddings,
    "position_embeddings": position_embeddings,
    "layernorm_weight": layernorm_weight,
    "layernorm_bias": layernorm_bias,
}

torch.save(weight_dict, "weight_dict.pth")
```

코드를 실행하면 `weight_dict.pth` 파일을 얻을 수 있습니다.

### 캘리브레이션 데이터셋 준비 (Prepare Calibration Dataset)

Massive Text Embedding Benchmark (MTEB)에서 관리하는 [한국어 STS 벤치마크 데이터셋](https://huggingface.co/datasets/mteb/KorSTS)을 사용하여 캘리브레이션 데이터셋을 생성할 것입니다.

먼저 데이터셋의 문장들을 토큰화하고, 미리 추출한 임베딩 가중치를 사용하여 임베딩합니다. 그 후, 임베딩된 텍스트를 numpy 파일로 저장합니다.

이 작업은 `prepare_calib.py`를 실행하여 수행할 수 있습니다. 코드를 실행하면 `calib` 디렉토리가 생성됩니다.

## 모델 컴파일 (Model Compilation)

캘리브레이션 데이터셋이 준비되면, 모빌린트 Qubee 컴파일러를 사용하여 모델을 컴파일할 수 있습니다.

```python
from qubee import mxq_compile
from qubee.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qubee.model_dict.parser.backend.torch.util import wrap_tensor
from transformers import BertModel, BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(
        "jhgan/ko-sbert-sts",
        trust_remote_code=True,
    )
    model = BertModel.from_pretrained("jhgan/ko-sbert-sts", trust_remote_code=True)
    model.eval()

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    feed_dict = {}
    for k, v in inputs.items():
        wrapped = wrap_tensor(k, v)
        wrapped.src_shape[1].set_dynamic()
        feed_dict[k] = wrapped
    set_attention_mask(feed_dict["attention_mask"], "padding_mask")

    mxq_compile(
        model=model,
        save_path="ko-sbert-sts.mxq",
        calib_data_path="./calib",
        backend="torch",
        feed_dict=feed_dict,
    )
```

`compile_mxq.py` 코드를 실행하면 `ko-sbert-sts.mxq` 파일을 얻게 됩니다.
