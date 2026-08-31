# Mask Generation Model Compilation

This tutorial explains how to compile a promptable segmentation model with Mobilint `qbcompiler`.

The example uses [SAM2 Hiera large](https://github.com/facebookresearch/sam2) from Meta. SAM2 is not a single feed-forward network: an image encoder runs once per image, and a lightweight mask decoder runs once per prompt. This tutorial compiles both into separate MXQ models and keeps the prompt encoder on the host.

> **Note**: The encoder and decoder take different routes to MBLT. The encoder goes through ONNX like the other compilation tutorials, in two explicit steps (`sam2_export_onnx.py` then `sam2_onnx_to_mblt.py`). The decoder cannot: the current parser rejects its hypernetwork matmul, so `sam2_decoder_to_mblt.py` parses it with the legacy parser and no decoder ONNX is produced. Step 1 covers both.

## Prerequisites

Before you begin, make sure you have:

- `qbcompiler` and the `mblt` Python package, from a matching pair. The encoder uses the current ONNX frontend and the decoder uses the legacy parser; both ship in the same wheel.
- Python 3.10 or later
- A CUDA GPU for calibration
- A local checkout of [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- A SA-V archive (`sav_train` chunk, or `sav_val.tar` / `sav_test.tar`), downloaded from Meta and prepared with `prepare_sav.py`; see [Step 0](#step-0-prepare-the-sa-v-calibration-source).
- `transformers==4.57.1`, which the parsing wrappers are pinned to
- `onnxruntime`, used by the encoder export for constant folding and `--verify`. The compiler environment usually ships `onnxruntime-gpu`, which provides the same module; do not install both.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

SAM2 itself is not on PyPI, so install it from the official repository:

```bash
git clone https://github.com/facebookresearch/sam2.git /workspace/sam2
pip install -e /workspace/sam2
```

If you prefer not to install it, clone it anywhere and pass the path with `--sam2-root` instead.

The SAM2 checkpoint is downloaded from Hugging Face on first use, so the calibration host needs network access or a warm Hugging Face cache.

> **Note**: `qbcompiler` rejects SAM2 decoder variants it cannot handle at compile time. The parsing wrapper requires the official Hiera decoder contract with `use_multimask_token_for_obj_ptr=True`.

## Overview

The workflow has four main steps:

0. **Prepare the calibration source**: Extract a SA-V subset with `prepare_sav.py`.
1. **Build the MBLT graphs**: Export the encoder to ONNX and compile it; parse the decoder directly with the legacy parser.
2. **Prepare the calibration dataset**: Generate encoder and decoder calibration tensors from SA-V.
3. **Compile the models**: Convert both MBLT graphs to `.mxq` using that calibration data.

## How SAM2 Is Split

The compiled models do not cover the whole network. Three stages stay on the host and run with official SAM2 code:

```text
image
  -> SAM2 image transform                     host
  -> image encoder                            encoder MXQ
  -> prompt encoder                           host
  -> mask decoder                             decoder MXQ
  -> mask upscaling                           host
```

The decoder MXQ takes the prompt encoder's raw outputs: the output-token concat and the `image_embeddings + dense_prompt_embeddings` sum happen inside its graph, in a host-side bridge subgraph the parser creates. Because the host still prepares those inputs, calibration must be produced by exactly the same host path that the runtime tutorial uses. `sam2_host.py` is that shared path, and it is duplicated in the runtime tutorial so each directory stays self-contained.

### Decoder Input Ordering

The decoder has six inputs and three of them share the shape `(1, 256, 64, 64)`, so array position alone is not enough to identify them. Calibration therefore records the MBLT input name and the semantic role of every slot.

Current MBLT input name order, used for calibration and compilation:

```text
image_embeddings            -> image_embeddings          (1, 256,  64,  64)
dense_prompt_embeddings     -> dense_prompt_embeddings   (1, 256,  64,  64)
image_pe                    -> image_pe                  (1, 256,  64,  64)
sparse_prompt_embeddings_0  -> sparse_prompt_embeddings  (1,   1,   N, 256)
high_res_features0_0        -> hrf0_nhwc                 (1, 256, 256,  32)
high_res_features1_0        -> hrf1_nhwc                 (1, 128, 128,  64)
```

`sparse_prompt_embeddings` carries the prompt axis `N`, which is one embedding per point plus one padding entry. If your decoder MBLT reports different input names, supply a new map with `--decoder-input-bindings` instead of assuming the defaults still apply.

## Step 0: Prepare the SA-V Calibration Source

Calibration frames and masks come from SA-V. **Download it yourself first**: the official [SA-V dataset guide](https://github.com/facebookresearch/sam2/blob/main/sav_dataset/README.md) documents the splits and points at the form-gated [downloads page](https://ai.meta.com/datasets/segment-anything-video-downloads/). This tutorial never fetches the data and does not work around that gate; obtain it from Meta directly rather than from a mirror.

Once you have a `.tar`, `prepare_sav.py` turns it into a calibration-ready subset:

```bash
python prepare_sav.py --archive sav_val.tar --videos 155
```

Both SA-V layouts are accepted and detected automatically:

| Split | Layout | Files |
| --- | --- | --- |
| `sav_train` | train | `{video}.mp4` beside `{video}_manual.json`, masks as RLE |
| `sav_val`, `sav_test` | vos | `JPEGImages_24fps/{video}/{frame}.jpg` beside `Annotations_6fps/{video}/{object}/{frame}.png` |

Only a subset is extracted. Calibration needs 32 encoder and 300 decoder samples, while a full `sav_val.tar` is 15 GB across 155 videos, so the script keeps `--videos` videos and `--frames-per-video` annotated frames each, discarding frames with no annotation. Extracting all 155 videos from `sav_val.tar` yields **307 MB** rather than 15 GB. Use `--dry-run` to see the selection and its size before extracting.

Whole annotated frames are kept, with every object mask on them, so decoder calibration can still balance across object sizes instead of being handed one arbitrary object per frame.

The script prints the exact next command, including skip values sized to the videos you actually have:

```bash
python prepare_calibration.py --stage both --defer-manifest \
  --sav-root ./data/sav --seed 1234 \
  --encoder-samples 32 --encoder-skip-videos 0 --encoder-max-videos 32 \
  --decoder-samples 60 --decoder-skip-videos 36 --decoder-max-videos 60
```

### One Split Covers Calibration and Evaluation

A single split serves both, because the skip offsets carve **disjoint video ranges** out of it: no video is shared between encoder calibration, decoder calibration, and evaluation. This is the same arrangement the original recipe used inside `sav_train`, just at a smaller scale.

`prepare_sav.py` prints the three ranges for the videos you actually have. For `sav_val.tar`'s 155 videos:

```text
disjoint video ranges (no video is shared between them):
  encoder calibration :   0 -  31
  decoder calibration :  36 -  95
  evaluation reserve  : 100 - 154  (55 videos)
```

The command it prints passes `--encoder-max-videos` and `--decoder-max-videos`, which make each range a **hard bound**. Those flags are not cosmetic: a range cannot be sized as `samples / per_video`, because a video may yield fewer samples than asked (`iter_frame_samples` builds a set of jittered frame indices that can collapse to one, and `build_prompt()` rejects masks too thin to place a point in). Without the bound the encoder set walks past its range and reaches into the decoder's, and the decoder set reaches into the evaluation reserve — silently, since nothing raises. The ranges are therefore sized for the worst case of one sample per video.

With the bound, running out of room fails loudly instead (`requested 32 encoder samples, wrote 4`), so a too-small range is visible rather than leaking. The order is shuffled under `--seed`, so pass the same `--seed` to `prepare_calibration.py` and the ranges stay reproducible.

What matters is that no video appears in two ranges; which split you draw them from does not. `sav_train` is the intended source when you can download it, but `sav_val` or `sav_test` works the same way.

### Video Counts

The default skip ranges assume the large `sav_train` split:

| Set | Arithmetic | Videos needed |
| --- | --- | ---: |
| Encoder | skip 600, then 32 samples / 2 per video | 616 |
| Decoder | skip 800, then 300 samples / 4 per video | 875 |

`sav_val` has 155 videos and `sav_test` has 150, so those defaults do not fit either one and `prepare_sav.py` prints smaller, still-disjoint ranges instead. The decoder also needs somewhat more videos than the arithmetic suggests, since `build_prompt()` returns `None` for masks too thin to place a point in and the selector yields fewer than `--decoder-per-video` on videos with few large objects.

> **Note**: video selection is shuffled under `--seed`, so extracting more videos later reshuffles the split and previously generated calibration no longer maps to the same videos. Finish preparing before generating, or regenerate both sets together.

## Step 1: Build the MBLT Graphs

The two halves take different routes, because they need different parsers.

```bash
python sam2_export_onnx.py            # encoder: SAM2 -> ONNX
python sam2_onnx_to_mblt.py           # encoder: ONNX -> MBLT
python sam2_decoder_to_mblt.py        # decoder: SAM2 -> MBLT
```

**The encoder** goes through ONNX, the same route the other compilation tutorials use. `sam2_export_onnx.py` hooks `image_encoder` during `set_image` and exports the Hiera trunk and FPN neck through `Sam2ImageEncoderWrapper`; `sam2_onnx_to_mblt.py` parses it with `mblt_compile(..., backend="onnx")`. The result reports one input, `input_image_channel_last`, and three FPN outputs.

**The decoder** cannot take that route: `mblt_compile` uses the current parser, whose matmul transform rejects SAM2's hypernetwork head with `unable to broadcast: 256, 32`. `sam2_decoder_to_mblt.py` routes it to the legacy parser (`qbcompiler.model_dict`), which lowers the same matmul without complaint. It parses `sam_mask_decoder` directly rather than a wrapper, and the parser splits the graph itself:

```text
subgraph 0  host bridge   11 ops   output-token concat, image_embeddings + dense
subgraph 1  NPU body     168 ops   TwoWayTransformer, upscaling, hypernetwork/IoU heads
```

That split is the same one the tutorial would otherwise perform by hand. The prompt axis is marked dynamic on `sparse_prompt_embeddings`, so one decoder serves any point count.

Outputs:

- `sam2_hiera_large_encoder.onnx` and `sam2_hiera_large_encoder.mblt`
- `sam2_hiera_large_decoder.mblt`

The default trace image is `../../runtime/python/rc/bus.jpg` and the default prompt is the three-point prompt in `sam2_host.prompt_arrays()`. Pass `--sam2-root` if your `sam2` checkout is not already importable.

### Constant Folding Is Not Optional

Hiera interpolates its position embedding to `x.shape`, which the ONNX exporter records as a live `Shape`/`Gather`/`Div` chain feeding a `Resize`. The parser cannot place that chain, so it cuts the graph there and the patch embed silently becomes a **second** graph input:

```text
inputs: ['input_image_channel_last', '/image_encoder/trunk/Transpose_output_0']
```

That graph parses without raising and is wrong. `sam2_export_onnx.py` therefore constant-folds every export with `onnxruntime` before writing it, which restores the single-input graph. `--no-fold` exists for debugging only.

Do not reach for the extended `onnxruntime` optimization level by hand. Its fusions emit ops the ONNX frontend does not convert, and the parse then falls back to a 23-op fragment of the first block.

The export also applies the same `qbcompiler` patcher the torch parser applies, so the exported graph carries the device-friendly rewrites rather than the stock ones. Pass `--no-patch` only to inspect the unpatched graph.

### Verifying the Export

`--verify` is on by default. It runs the exported encoder under `onnxruntime` and compares every output against the torch outputs the export traced. This script exports only the encoder, so it makes no claim about the decoder; the decoder's dynamic prompt axis is established by `sam2_decoder_to_mblt.py` and confirmed on the compiled artifact with `qbruntime.get_model_summary`, which reports the prompt axis as `-1`.

The comparison is relative to each output's magnitude, not absolute. `onnxruntime` runs on the CPU, so a CUDA export is compared across devices. The export disables TF32 for this reason: on Ampere and later GPUs the default TF32 mantissa drifts the reference forward by roughly `2e-3` relative across the Hiera trunk, which is enough to fail an honest tolerance even though the export is exact.

### Trace Memory

Exporting evaluates every operation eagerly, so the Hiera-large encoder needs more than 12 GB of VRAM. On a 12 GB card it fails with `torch.OutOfMemoryError` partway through the trunk. Export on the CPU instead:

```bash
python sam2_export_onnx.py --part encoder --torch-device cpu
```

The encoder export takes about two minutes on the CPU and roughly 10 GB of RAM. `--torch-device` defaults to CUDA when it is available and otherwise falls back to the CPU.

### The Dynamic Prompt Axis

A trace pins the prompt length to whatever prompt it captured, which would give a decoder that only ever accepts that one prompt size. That rules out the interactive loop image segmentation is for: click, look, add a corrective click. `sam2_decoder_to_mblt.py` marks the axis dynamic instead:

```python
feed["sparse_prompt_embeddings"].src_shape[-2].set_dynamic(True)
```

The parsed graph then reports `sparse_prompt_embeddings_0` as `(1, 1, dyn(4), 256)`. Step 2 records the same constraint in the calibration manifest as `shapes_by_role["sparse_prompt_embeddings"][2] = -1`.

### Why the Decoder Uses the Legacy Parser

`mblt_compile` and `sam2_export_onnx.py` both go through the current parser (`qbcompiler.model_dict_new`), whose matmul transform rejects the decoder's hypernetwork head:

```text
mblt/transform/ruleset/dataflow/matmul.py, in case0
ValueError: unable to broadcast: 256, 32
```

That is `hyper_in @ upscaled_embedding`, and it fails on the ONNX and torch frontends alike because the blocker is in `mblt-graph`'s transform rules rather than in either frontend. The legacy parser (`qbcompiler.model_dict`) has its own lowering path and handles the same matmul, which is why the decoder is routed there. `qbcompiler/scripts/sam2/sam2_devel_decoder.py` is the reference this follows.

When a `qbcompiler`/`mblt-graph` pair that includes the SAM2 decoder changes becomes available, the decoder can move to the ONNX route and match the encoder. The input contract would change with it, so re-run Step 1-1 if you switch.

## Step 1-1: Confirm the Decoder Input Names

The decoder input names depend on the parsed graph, and Step 2 and Step 3 both rely on them. Print the names your MBLT actually reports:

```bash
python -c "from decoder_bindings import read_model_input_names; print(read_model_input_names('./sam2_hiera_large_decoder.mblt'))"
```

Compare the result with `decoder_input_bindings.json`. If the names differ, write a new map from those names to the same six semantic roles and pass it with `--decoder-input-bindings` in Step 2 and Step 3. Do not reorder the calibration tensors to fit the defaults: several decoder inputs share a shape, so a positional guess fails silently rather than raising.

## Step 2: Prepare the Calibration Dataset

`prepare_calibration.py` builds both calibration sets from SA-V manual masklets. Encoder calibration needs only frames, while decoder calibration also needs a ground-truth mask so a point prompt can be placed inside the object.

Generate both sets in one run:

```bash
python prepare_calibration.py --sav-root ./data/sav --decoder-model ./sam2_hiera_large_decoder.mblt \
  --encoder-samples 32 --encoder-skip-videos 0 --encoder-max-videos 32 \
  --decoder-samples 60 --decoder-skip-videos 36 --decoder-max-videos 60
```

Outputs, written next to the scripts rather than into the current working directory, so the tutorial's `calib/` tree fills in the same place no matter where you run from:

- `calib/encoder/encoder_calib.txt`: listing of encoder tensor paths
- `calib/encoder/encoder/*.npy`: encoder tensors
- `calib/decoder/decoder_calib.json`: decoder manifest with input names, slot roles, and tensor paths
- `calib/decoder/decoder/<role>/*.npy`: decoder tensors, one directory per semantic role

`model_compile.py` reads the same two paths by default. Override either side with `--encoder-output-dir` / `--decoder-output-dir` and the matching `--encoder-calib` / `--decoder-calib`.

The encoder and decoder sets are drawn from disjoint video ranges (`--encoder-skip-videos 600` and `--decoder-skip-videos 800`) so the two models are not calibrated on the same footage. You can also generate one set at a time with `--stage encoder` or `--stage decoder`.

### Encoder Calibration

Each encoder sample is the output of the official SAM2 transform: a float32 NHWC tensor of shape `[1, 1024, 1024, 3]`. The script fails immediately if a sample has any other shape, because a silent preprocessing change would quantize the encoder against the wrong input distribution.

### Decoder Calibration and the Dynamic Prompt Axis

The prompt encoder emits one embedding per point plus one padding entry, so a prompt with `N` points produces `N + 1` entries. The default `--point-mix 1,2,3` cycles through one, two, and three point prompts, which yields prompt lengths of 2, 3, and 4. The 6 output tokens SAM2 prepends are concatenated inside the decoder graph now, so they no longer appear in what the host hands over.

When the point mix contains more than one distinct value, the manifest marks the prompt axis as dynamic:

```python
if len(set(points_per_sample)) > 1:
    shapes_by_role["sparse_prompt_embeddings"][2] = -1
```

Without this the compiled decoder is fixed to a single prompt length and rejects prompts with a different number of points. Keep the default mix unless you intend to support only one prompt size.

### Deferring the Decoder Manifest

The decoder manifest is keyed by the input names the quantizer sees, and those are the **post-parse** names, not the names in an ONNX file. `--decoder-model` therefore accepts either form and always resolves the contract the same way the compile will:

- a `.mblt` is read directly, as before;
- a `.onnx` from `sam2_export_onnx.py` is parsed with the same parser configuration the compile uses (weights skipped), and the resulting graph's input names are used.

Decoder tensor generation does not need the model at all: the tensors come from the official FP32 host path and are keyed by semantic role, not by model input name. Only the manifest needs the decoder MBLT, to record its input names. `--defer-manifest` splits the two, which is useful when the tensors are ready before a parseable decoder is:

```bash
python prepare_calibration.py --stage decoder --defer-manifest --sav-root ./data/sav
python prepare_calibration.py --stage manifest --decoder-model ./sam2_hiera_large_decoder.mblt
```

This writes the role-keyed tensors plus `calib/decoder/decoder_tensor_meta.json`, then the manifest, without re-running SA-V or the FP32 encoder in between. Encoder calibration needs no model file at all.

## Step 3: Compile the Models

`model_compile.py` calls `mxq_compile` once for the encoder and once for the decoder. Before compiling, it re-reads the decoder MBLT and refuses to continue if the calibration manifest was generated against a different graph:

```python
if info.get("input names") != model_inputs:
    raise ValueError(
        "decoder calibration input names do not match the MBLT. Regenerate calibration "
        "with this exact decoder MBLT instead of relying on positional same-shape inputs."
    )
```

This check exists because a positional mismatch between same-shape inputs would quantize the wrong tensors and still produce a compiled model that runs.

Compilation settings come from `compile_config.json` rather than inline arguments:

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

`weightMemory.method` and `useGPUOnlyForCalibration` are `CompileConfig` fields with no `mxq_compile` keyword equivalent, so they are loaded with `CompileConfig.from_file` instead.

Like the other tutorials, `model_compile.py` uses CUDA when `torch.cuda.is_available()` is true and otherwise falls back to CPU, printing the selected host device before compilation starts.

Run the compilation:

```bash
python model_compile.py --target-device aries-rb
```

`--part` selects which models to compile. While the decoder cannot be parsed, compile the encoder on its own; the decoder manifest check is skipped along with it:

```bash
python model_compile.py --part encoder
```

Add `--dry-run` to validate the files, the manifest, and the MBLT input contract without compiling.

Outputs:

- `sam2_hiera_large_encoder.mxq`
- `sam2_hiera_large_decoder.mxq`

The encoder MXQ is about 268 MB, down from the 848 MB MBLT.

The runtime tutorial in [../../runtime/python/mask_generation/README.md](../../runtime/python/mask_generation/README.md) expects both files at these exact paths.

### The Decoder Compiles on the CPU

The decoder MBLT has two subgraphs, so `mxq_compile` requires `cpu_offload=True`; with the default `cpu_offload=False` it refuses outright:

```text
ValueError: cpu_offload=False cannot compile a model with 2 subgraphs
```

Its host bridge also concatenates a CPU-resident constant with the calibration tensors, which fails mid-quantization on the GPU:

```text
RuntimeError: Error. quantizeFB failed. Expected all tensors to be on the same device,
but found at least two devices, cpu and cuda:0!
```

`model_compile.py` therefore compiles the decoder with `cpu_offload=True` and `device="cpu"`, while the encoder keeps the faster GPU path. This is automatic; no flag is needed.

### Select the Target Device (`--target-device`)

| User | `--target-device` | Model |
| --- | --- | --- |
| ARIES | `aries-rb` | `sam2-hiera-large` |

> **Note**: Only `aries-rb` has been validated for SAM2. REGULUS is not covered by this tutorial.

## Parameters

`prepare_calibration.py`:

- `--stage`: which set to generate, `encoder`, `decoder`, or `both`. Default: `both`.
- `--sav-root`: path to the SA-V `sav_train` directory. Required.
- `--sam2-root`: local `facebookresearch/sam2` checkout.
- `--model-id`: SAM2 model id. Default: `facebook/sam2-hiera-large`.
- `--encoder-samples`: number of encoder samples. Default: `32`.
- `--decoder-samples`: number of decoder samples. Default: `300`.
- `--point-mix`: point counts cycled across decoder samples. Default: `1,2,3`.
- `--encoder-skip-videos`, `--decoder-skip-videos`: shuffled videos to skip, keeping the two sets disjoint. Defaults: `600` and `800`.
- `--decoder-model`: decoder `.mblt` or `.onnx` whose post-parse input names the manifest must match.
- `--defer-manifest`: generate decoder tensors without emitting the manifest.
- `--stage manifest`: emit the manifest from previously saved tensors and `--decoder-model`.
- `--decoder-input-bindings`: MBLT input name to semantic role map. Default: `./decoder_input_bindings.json`.

`model_compile.py`:

- `--encoder-mblt`, `--decoder-mblt`: input MBLT graphs.
- `--encoder-calib`, `--decoder-calib`: calibration listing and manifest from Step 2.
- `--encoder-save-path`, `--decoder-save-path`: MXQ output paths.
- `--compile-config`: `CompileConfig` JSON. Default: `./compile_config.json`.
- `--target-device`: target NPU. Default: `aries-rb`.
- `--gpu`: CUDA device index used for calibration. Default: `0`.
- `--part`: which models to compile, `encoder`, `decoder`, or `both`. Default: `both`.
- `--dry-run`: validate inputs without compiling.

## Validated Result

This calibration recipe was measured on 200 one-click SA-V samples drawn from videos disjoint from the calibration ranges, running both MXQ models on an Aries2 NPU with `qbruntime` v1.2.0:

| Path | Samples | mIoU |
| --- | ---: | ---: |
| Official FP32 | 200 | 0.775005 |
| Encoder + decoder MXQ | 200 | 0.775706 |

Binary mask agreement against FP32 was `0.983084` and low-resolution logit cosine similarity was `0.998363`. The compiled decoder accepted all three calibrated token lengths (8, 9, and 10).

Reproducing these numbers requires the evaluation harness, which is outside the scope of this tutorial.

These figures were measured on a decoder MBLT from an earlier Mobilint model build, which exposed four outputs. Depending on your `qbcompiler` build, the decoder parsed in Step 1 may expose two outputs (masks and IoU) or four (adding SAM tokens and object score). The runtime tutorial handles both, and only masks and IoU affect the segmentation result.

## Files in This Tutorial

- `prepare_sav.py`: extracts a calibration-ready subset from a downloaded SA-V archive
- `sam2_export_onnx.py`: exports the encoder wrapper to ONNX
- `sam2_onnx_to_mblt.py`: compiles the encoder ONNX to MBLT with `mblt_compile`
- `sam2_decoder_to_mblt.py`: parses the mask decoder to MBLT with the legacy parser
- `prepare_calibration.py`: generates encoder and decoder calibration data from SA-V
- `model_compile.py`: compiles both MBLT graphs into MXQ for the selected `--target-device`
- `sam2_host.py`: host-side SAM2 helpers shared by parsing and calibration generation
- `sav_dataset.py`: SA-V frame, mask, and point-prompt sampling
- `decoder_bindings.py`: MBLT input name to semantic role resolution
- `decoder_input_bindings.json`: default binding map
- `compile_config.json`: `qbcompiler` `CompileConfig` used for both models
- `requirements.txt`: Python dependencies for calibration generation
- `README.md`: documents the end-to-end workflow for this example
