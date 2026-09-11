# Mask Generation Model Compilation

This tutorial compiles the image encoder and mask decoder from Meta's [SAM2 Hiera large](https://github.com/facebookresearch/sam2) into separate MXQ models. Image preprocessing, prompt encoding, decoder input preparation, and mask upscaling remain on the host.

The encoder and decoder do not use ONNX. Each `compile_*.py` script creates an MBLT with `mblt_compile()` and then runs `mxq_compile()` in the same script.

## Supported Devices

| Device | Support |
| --- | --- |
| `aries-rb` | Supported |
| `regulus-rb` | Supported |
| `regulus-ra` | Not supported |

The default target is `aries-rb`.

## Prerequisites

- Python 3.10 or later
- `transformers==5.16.1`
- [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- An SA-V archive downloaded by the user

Install the tutorial dependencies.

```bash
pip install -r requirements.txt
```

Install SAM2 from its official repository.

```bash
git clone https://github.com/facebookresearch/sam2.git /workspace/sam2
pip install -e /workspace/sam2
```

## 1. Prepare SA-V Data

Download SA-V by following the official [dataset guide](https://github.com/facebookresearch/sam2/blob/main/sav_dataset/README.md). `prepare_sav.py` does not download the dataset. It extracts only the subset needed for calibration from an existing tar archive.

The default archive name is `sav_val.tar`.

```bash
python prepare_sav.py
```

Pass the path when the archive has another name.

```bash
python prepare_sav.py --archive /path/to/sav_000.tar
```

The defaults prepare 120 videos under `./data/sav`. Encoder calibration, decoder calibration, and evaluation reserve use disjoint video ranges.

| Purpose | Video positions |
| --- | --- |
| Encoder calibration | 0–31 |
| Decoder calibration | 36–95 |
| Evaluation reserve | 100 onward |

Using the same `--seed` reproduces the video and sample selection.

## 2. Generate Calibration Data

Generate both encoder and decoder calibration tensors.

```bash
python prepare_calibration.py
```

Either set can be generated separately.

```bash
python prepare_calibration.py --stage encoder
python prepare_calibration.py --stage decoder
```

The main outputs are:

```text
calib/encoder/encoder_calib.txt
calib/encoder/encoder/*.npy
calib/decoder/decoder_tensor_meta.json
calib/decoder/decoder/<role>/*.npy
```

Encoder inputs are float32 NHWC `[1, 1024, 1024, 3]` tensors produced by the official SAM2 transform. Decoder calibration cycles through one-, two-, and three-point prompts by default.

## 3. Compile the Encoder

Compile the encoder for ARIES.

```bash
python compile_encoder.py --target-device aries-rb
```

Compile it for REGULUS with:

```bash
python compile_encoder.py --target-device regulus-rb
```

The script performs these steps:

1. Capture an actual `predictor.set_image()` input.
2. Create `mblt/<target-device>/sam2_hiera_large_encoder.mblt`.
3. Create `mxq/<target-device>/sam2_hiera_large_encoder.mxq` with the encoder calibration data.

## 4. Compile the Decoder

Compile the decoder for ARIES.

```bash
python compile_decoder.py --target-device aries-rb
```

Compile it for REGULUS with:

```bash
python compile_decoder.py --target-device regulus-rb
```

The script performs these steps:

1. Capture an actual `predictor.predict()` input.
2. Create `mblt/<target-device>/sam2_hiera_large_decoder.mblt`.
3. Read the generated MBLT input names and create `calib/decoder/decoder_calib.json`.
4. Create `mxq/<target-device>/sam2_hiera_large_decoder.mxq` with the decoder calibration data.

The decoder `tokens` axis is dynamic, allowing the one- to three-point prompts included in calibration.

## Decoder Input Contract

The decoder has six inputs, and three of them have the same shape. The calibration manifest therefore maps MBLT input names to semantic roles instead of guessing by position.

```text
tokens/reshape                  -> tokens        (1, 1,    T, 256)
add/transpose                   -> src_plus_pos  (1, 1, 4096, 256)
flatten/reshape/transpose       -> src           (1, 1, 4096, 256)
flatten_1/reshape/transpose     -> pos_src       (1, 1, 4096, 256)
high_res_features_1/transpose  -> hrf1_nhwc     (1, 128, 128, 64)
high_res_features_0/transpose  -> hrf0_nhwc     (1, 256, 256, 32)
```

The mapping is stored in `decoder_input_bindings.json`. Because `compile_decoder.py` reads the MBLT it just created before writing the manifest, a stale manifest cannot be paired with a different graph.

## Outputs

Selecting `aries-rb` creates:

```text
mblt/aries-rb/sam2_hiera_large_encoder.mblt
mblt/aries-rb/sam2_hiera_large_decoder.mblt
mxq/aries-rb/sam2_hiera_large_encoder.mxq
mxq/aries-rb/sam2_hiera_large_decoder.mxq
```

Selecting `regulus-rb` creates the same files under `mblt/regulus-rb` and `mxq/regulus-rb`.

## Main Options

`compile_encoder.py` and `compile_decoder.py`:

- `--target-device`: `aries-rb` or `regulus-rb`. Default: `aries-rb`.
- `--model-id`: Hugging Face SAM2 model ID.
- `--image`: Image used to capture the MBLT inputs.
- `--device`: Torch device for the host SAM2 model. Default: `cuda`, with CPU used when CUDA is unavailable.

`prepare_calibration.py`:

- `--stage`: `encoder`, `decoder`, or `both`. Default: `both`.
- `--sav-root`: Extracted SA-V root. Default: `./data/sav`.
- `--encoder-samples`: Number of encoder samples. Default: 32.
- `--decoder-samples`: Number of decoder samples. Default: 60.
- `--point-mix`: Decoder point counts. Default: `1,2,3`.
- `--seed`: Sample selection seed. Default: 1234.

## Files

- `prepare_sav.py`: Extracts the calibration subset from an SA-V tar archive.
- `prepare_calibration.py`: Generates encoder and decoder calibration tensors.
- `compile_encoder.py`: Creates the encoder MBLT and MXQ.
- `compile_decoder.py`: Creates the decoder MBLT, calibration manifest, and MXQ.
- `sam2_host.py`: Shared SAM2 host processing for calibration and compilation.
- `decoder_bindings.py`: Maps decoder MBLT input names to semantic roles.
- `compile_config.json`: MXQ compilation settings.

See the [runtime tutorial](../../runtime/python/mask_generation/README.md) for inference.
