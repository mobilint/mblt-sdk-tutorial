# Large Language Model (LLM) Compilation

This tutorial compiles [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) with Mobilint `qbcompiler` and prepares it for runtime inference.

## Prerequisites

- `qbcompiler`
- A Hugging Face account with access to the Llama model
- Optional CUDA GPU

```bash
pip install -r requirements.txt
huggingface-cli login --token <your_huggingface_token>
```

## 1. Generate Calibration Data

Calibration uses Wikipedia text from all eight languages officially supported by Llama 3.2: English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.
The default 128 samples are distributed evenly across those languages.
`--languages` changes the calibration language list and does not restrict the runtime language.
Korean input is possible but is not officially supported by the model.

```bash
python generate_calib.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --output-dir ./calibration_data
```

## 2. Compile W8

```bash
python mxq_compile.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --calib-data-path ./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/multilingual \
  --save-path ./Llama-3.2-1B-Instruct-W8.mxq \
  --target-device aries-rb
```

`--target-device` accepts `aries-rb` and `regulus-rb`.
For REGULUS, only `regulus-rb` is supported; `regulus-ra` is not supported.

## 3. Prepare the Runtime Model

`prepare_models.py` downloads the runtime files from the Mobilint Hugging Face repository and replaces only the MXQ file and its path in `config.json`.

```bash
python prepare_models.py \
  --mxq-path ./Llama-3.2-1B-Instruct-W8.mxq \
  --output-folder ./llama-mxq-w8 \
  --revision W8
```

Continue with the [LLM runtime tutorial](../../runtime/python/llm/README.md).

## Optional: W4V8

```bash
python mxq_compile_4bit.py \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --calib-data-path ./calibration_data/datas/meta-llama-Llama-3.2-1B-Instruct/multilingual \
  --save-path ./Llama-3.2-1B-Instruct-W4V8.mxq \
  --target-device aries-rb

python prepare_models.py \
  --mxq-path ./Llama-3.2-1B-Instruct-W4V8.mxq \
  --output-folder ./llama-mxq-w4v8 \
  --revision W4V8
```
