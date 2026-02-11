# Bidirectional Encoder Representations from Transformers (BERT)

이 튜토리얼은 Mobilint qb 컴파일러를 사용하여 BERT 모델을 컴파일하는 세부 지침을 제공합니다. 컴파일 프로세스는 표준 BERT 모델을 Mobilint NPU 하드웨어에서 효율적으로 실행할 수 있는 최적화된 `.mxq` 형식으로 변환합니다.

이 튜토리얼에서는 BERT 아키텍처를 기반으로 하고 문장 임베딩을 생성하도록 수정된 [Sentence-BERT](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) 모델을 사용합니다.

## 사전 요구 사항

시작하기 전에 다음이 설치되어 있는지 확인하십시오.

- Mobilint qb 컴파일러 (버전 1.0.0 이상 필요)
- CUDA 지원 GPU (컴파일 시간 단축을 위해 권장)

또한, 다음 패키지를 설치해야 합니다.

```bash
pip install accelerate datasets
```

## 모델 분석

BERT의 복잡한 아키텍처로 인해 일부 레이어는 컴파일러에서 지원되지 않을 수 있습니다. 따라서 컴파일 프로세스 전에 모델 분석을 수행해야 합니다.

모델 구조를 분석하기 위해 모델을 MBLT 형식으로 변환합니다.

```python
from qbcompiler import mblt_compile
from qbcompiler.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
from transformers import BertModel, BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        trust_remote_code=True,
    )
    model = BertModel.from_pretrained("sentence-transformers-testing/stsb-bert-tiny-safetensors", trust_remote_code=True)
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
        mblt_save_path="stsb-bert-tiny-safetensors.mblt",
        backend="torch",
        feed_dict=feed_dict,
        cpu_offload=True,
    )
```

`compile_mblt.py`를 실행하면 `stsb-bert-tiny-safetensors.mblt` 파일을 얻을 수 있습니다. [Netron](https://netron.mobilint.com)을 사용하여 모델 아키텍처를 시각화할 수 있습니다.

Netron 시각화 결과에 따르면, 일부 입력 임베딩 레이어는 지원되지 않습니다. 따라서 지원되는 레이어의 입력에 맞추기 위해 캘리브레이션 데이터셋을 준비하고, 지원되지 않는 연산은 Mobilint NPU 외부에서 실행해야 합니다.

## 캘리브레이션 데이터셋 준비

### 임베딩 가중치 추출

Netron에서 볼 수 있듯이 입력 임베딩 부분은 지원되지 않습니다. 따라서 지원되는 레이어의 입력에 맞게 캘리브레이션 데이터셋을 준비해야 합니다. 이를 위해 모델에서 임베딩 가중치를 추출하여 `.pth` 파일로 저장합니다.

이 작업은 `get_embedding.py`를 실행하여 수행할 수 있습니다.

```python
import torch

from transformers import BertModel

model = BertModel.from_pretrained(
    "sentence-transformers-testing/stsb-bert-tiny-safetensors", trust_remote_code=True
)

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

스크립트를 실행하면 `weight_dict.pth` 파일을 얻을 수 있습니다.

### 캘리브레이션 데이터 생성

Massive Text Embedding Benchmark (MTEB)에서 관리하는 [STS Benchmark Dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts)을 사용하여 캘리브레이션 데이터셋을 생성합니다.

먼저 데이터셋의 문장을 토큰화하고 이전에 추출한 임베딩 가중치를 사용하여 임베딩합니다. 그런 다음 임베딩된 텍스트를 NumPy 파일로 저장합니다.

이 작업은 `prepare_calib.py`를 실행하여 수행할 수 있습니다. 실행 후 `calib` 디렉토리가 생성됩니다.

## 모델 컴파일

캘리브레이션 데이터셋이 준비되면 Mobilint qb 컴파일러를 사용하여 모델을 컴파일할 수 있습니다.

```python
from qbcompiler import mxq_compile, QuantizationConfig
from qbcompiler.model_dict.parser.backend.torch.object_wrapper import set_attention_mask
from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor

from transformers import BertModel, BertTokenizer

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        trust_remote_code=True,
    )
    model = BertModel.from_pretrained(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", trust_remote_code=True
    )
    model.eval()

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    feed_dict = {}
    for k, v in inputs.items():
        wrapped = wrap_tensor(k, v)
        wrapped.src_shape[1].set_dynamic()
        feed_dict[k] = wrapped
    set_attention_mask(feed_dict["attention_mask"], "padding_mask")

    quantization_config = QuantizationConfig.from_kwargs(
        quantization_method=1,  # 0 for per tensor, 1 for per channel
        quantization_output=0,  # 0 for layer, 1 for channel
        quantization_mode=1,  # maxpercentile
        percentile=0.999,  # quantization percentile
        topk_ratio=0.01,  # quantization topk
    )

    mxq_compile(
        model=model,
        save_path="stsb-bert-tiny-safetensors.mxq",
        calib_data_path="./calib",
        backend="torch",
        feed_dict=feed_dict,
        quantization_config=quantization_config,
    )

```

`compile_mxq.py`를 실행하면 `stsb-bert-tiny-safetensors.mxq` 파일을 얻을 수 있습니다.
