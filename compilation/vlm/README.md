# Vision Language Model (VLM) Compilation

This tutorial explains how to compile a vision-language model (VLM) with Mobilint `qbcompiler`.

In this tutorial, we will use the [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) model, a state-of-the-art vision-language model developed by Qwen. Qwen3-VL introduces a deepstack visual pathway: the vision encoder produces image embeds plus three deepstack feature maps that are injected into the early decoder layers, so both the calibration data and the decoder compilation carry deepstack tensors.

The quantization settings in this tutorial use the benchmark-best 4B configuration: the decoder is compiled with 4-bit weights (value projections promoted to 8-bit), 16-bit activations on the embedding and deepstack inputs, SpinR1/SpinR2 rotations, OPTQ, and a weight-scale search; the encoder uses 16-bit activations on the merger and deepstack merger `fc2` layers together with a `head_out_ch_rotation` that reuses the decoder's SpinR1 matrix.

## Overview

The VLM compilation process consists of three main stages:

1. **Calibration Data Generation**: Create calibration datasets for quantization.
2. **MBLT Compilation**: Compile the model to MBLT (Mobilint Binary LayouT) format.
3. **MXQ Compilation**: Apply advanced quantization and compile to `.mxq` format for deployment.

The workflow compiles the **language model** (decoder) and the **vision encoder** separately.

After compilation, the `mxq/` directory contains all files required for deployment on the NPU.

## Prerequisites

Before you begin, make sure the following are available:

- Python 3.8 or higher
- `qbcompiler` SDK installed (version `>= 1.0.1`)
- Optional CUDA-capable GPU for calibration and compilation
- Sufficient disk space (about 20 GB for the model and calibration data)

### Install Required Dependencies

Install the required Python packages for compilation:

```bash
pip install transformers==4.57.1 qwen-vl-utils==0.0.14 accelerate==1.13.0
```

### Download Calibration Images

The calibration flow uses images from the COCO dataset. A helper script is provided to download 100 images automatically:

```bash
python download_images.py
```

**What it does:**

- Downloads 100 images from the COCO 2017 validation set using Hugging Face Datasets
- Automatically resizes the images to `224x224`
- Saves the images as JPEG files in the `images/` directory
- Falls back to synthetic sample images if the COCO download fails

**Output:**

- `images/image_0000.jpg` through `images/image_0099.jpg`

The calibration scripts will automatically use all images in the `images/` directory and cycle through diverse prompts (detailed descriptions, visual reasoning, counting, spatial understanding, etc.) to ensure calibration diversity.

## Stage 1: Calibration Data Generation

Calibration data is essential for quantization, as it helps the compiler understand the typical activation ranges of the model.

### Step 1.1: Generate Calibration Data

A single script generates calibration data for both the language model (decoder) and the vision encoder while reusing one loaded model:

```bash
python generate_calibration_data.py \
    --model-name Qwen/Qwen3-VL-4B-Instruct \
    --output-dir ./calibration_data \
    --num-samples 100 \
    --max-new-tokens 512
```

**Parameters:**

- `--model-name`: HuggingFace model identifier
- `--output-dir`: Base directory; `language/`, `vision/`, `prefill/`, and `decode/` subdirectories are created under it
- `--num-samples`: Number of calibration samples (default: all available images)
- `--max-new-tokens`: Maximum tokens for the decode generation pass
- `--intermediate-ratios`: Decode-prefix ratios to save (1.0 is always appended)

**What it does:**

- Loads all images from `images/` folder and cycles through 20 diverse prompt types
- Language: runs two passes per image. Pass 1 captures the decoder prefill inputs (`inputs_embeds` and the three `deepstack_visual_embeds` layers) after vision features are merged; Pass 2 runs a full generate to collect the decode token sequence. Prefill and decode samples are then merged into a single `language/` directory
- Vision: captures vision encoder pixel values reshaped to the NPU layout `[H, W, 6]` (image size fixed at 224x224)
- Saves calibration data as `.npy` files; `language/` is indexed by a single `npy_files.json`

**Output structure:**

```text
calibration_data/
 language/
    prefill_000/{inputs_embeds.npy, deepstack_visual_embeds.npy}  # [1, seq_len, 2048], [3, seq_len, 2048]
    decode_000/{inputs_embeds.npy, deepstack_visual_embeds.npy}
    ...
    npy_files.json
 vision/
    sample_000/images.npy           # [H, W, 6]
    ...
    npy_files.txt
```

## Stage 2: MBLT Compilation

MBLT (Mobilint Binary LayouT) is an intermediate format that represents the model graph and weights in a hardware-agnostic way.

### Step 2.1: Compile Language Model to MBLT

Compile the language model (decoder) to MBLT format. `--target-device` is required (`aries-rb` or `regulus-rb`):

```bash
# ARIES
python mblt_compile_language.py --target-device aries-rb

# REGULUS (customers from 2026-06)
python mblt_compile_language.py --target-device regulus-rb
```

**What it does:**

- Captures language model inputs during sample generation
- Marks sequence length dimensions as dynamic (for variable-length inputs)
- Applies NPU-compatible architectural patches:
  - **Pre-cached RoPE embeddings**: Eliminates runtime trigonometric operations
  - **Last-query slicing**: Optimizes the final decoder layer for decode phase
  - **Stateful KV cache wrappers**: Enables efficient auto-regressive generation
  - **Dynamic shape handling**: Supports variable sequence lengths
- Pads the three `deepstack_visual_embeds` layers to the full sequence length via `build_full_visual_embeds` and `visual_pos_masks`
- Slices `position_ids` to the first 3 mrope axes (t/h/w) for the cached rotary embedding
- Configures dynamic shapes for attention operators
- Exports to MBLT format via `mblt_compile()`

**Key transformations:**

- Input embeddings dimension marked as dynamic: `[batch, seq_len, hidden_size]`
- `deepstack_visual_embeds` sequence-length axis marked as dynamic
- Attention mask and position IDs marked as dynamic for variable sequences
- Cache position marked as dynamic for auto-regressive generation
- RoPE embeddings pre-computed from the captured position IDs

**Output files:**

- `./mblt/Qwen3-VL-4B-Instruct_text_model.mblt`: Compiled model in MBLT format

### Step 2.2: Compile Vision Encoder to MBLT

Compile the vision encoder to MBLT format. `--target-device` is required (`aries-rb` or `regulus-rb`):

```bash
# ARIES
python mblt_compile_vision.py --target-device aries-rb

# REGULUS (customers from 2026-06)
python mblt_compile_vision.py --target-device regulus-rb
```

**What it does:**

- Captures vision encoder inputs during sample inference
- Reprocesses pixel values to NPU-compatible format
- Applies NPU-compatible architectural patches:
  - **3D2D convolution**: Transforms 3D convolutions to 2D for NPU optimization
  - **Split QKV projection**: Separates Query, Key, Value projections for better parallelization
  - **Pre-computed RoPE embeddings**: Eliminates runtime trigonometric operations
  - **Merged patchify operation**: Reduces memory transfers
- Exports to MBLT format via `mblt_compile()`

**Key transformations:**

- Pixel values reprocessed from HuggingFace format `[num_patches, channels*patch_size^2]` to ARIES format `[batch, channels*temporal, height, width]`
- 3D temporal convolutions converted to 2D spatial convolutions
- QKV attention projections split for parallel execution
- RoPE embeddings pre-computed based on image grid dimensions

**Output files:**

- `./mblt/Qwen3-VL-4B-Instruct_vision_transformer.mblt`: Compiled model in MBLT format

## Stage 3: MXQ Compilation (Advanced Quantization)

MXQ (Mobilint eXeQutable) format applies advanced quantization techniques and prepares the model for deployment on NPU.

### Step 3.1: Compile Language Model to MXQ

Compile the language model from MBLT to MXQ format. `--target-device` is required (`aries-rb` or `regulus-rb`):

```bash
# ARIES
python mxq_compile_language.py --target-device aries-rb

# REGULUS (customers from 2026-06)
python mxq_compile_language.py --target-device regulus-rb
```

**What it does:**

- Loads the MBLT file: `./mblt/Qwen3-VL-4B-Instruct_text_model.mblt`
- Loads calibration data from: `./calibration_data/language/npy_files.json`
- Applies advanced quantization with equivalent transformations
- Configures 16-bit activations for the embedding and deepstack inputs: `inputs_embeds/reshape`, `deepstack_visual_embeds_0`
- NPU inference scheme: set automatically by `--target-device` (`all` for ARIES, `single` for REGULUS)
- **Generates rotation matrix** at: `./spinWeight/Qwen3-VL-4B-Instruct_text_model/R1/global_rotation.pth`
  - This rotation matrix is **required for vision encoder MXQ compilation**

**Key configurations (benchmark-best 4B decoder):**

- Calibration: mode 0 (Max), output 0
- Weights: 4-bit (query/key/output/ffn/head), value projection promoted to 8-bit; float32 weight dtype during compilation
- Activation 16-bit layers: `["inputs_embeds/reshape", "deepstack_visual_embeds_0"]`
- Inference scheme: `all` for ARIES, `single` for REGULUS
- Equivalent transformations: UD (smoothing_factor=0.8), VO, SpinR1, SpinR2, optimize_ffn (QK disabled)
- OPTQ enabled (act_order, block_size=128, perc_damp=0.01)
- Weight-scale search enabled for query/key/value/out/ffn

**Output files:**

- `./mxq/Qwen3-VL-4B-Instruct_text_model.mxq`: Quantized model ready for deployment
- `./spinWeight/Qwen3-VL-4B-Instruct_text_model/R1/global_rotation.pth`: Global rotation matrix (needed for vision encoder)

### Step 3.2: Compile Vision Encoder to MXQ

**Important:** You must complete Step 3.1 (language model MXQ compilation) first, as the vision encoder compilation requires the rotation matrix generated during language model compilation.

Compile the vision encoder from MBLT to MXQ format. `--target-device` is required (`aries-rb` or `regulus-rb`):

```bash
# ARIES
python mxq_compile_vision.py --target-device aries-rb

# REGULUS (customers from 2026-06)
python mxq_compile_vision.py --target-device regulus-rb
```

**What it does:**

- Loads the MBLT file: `./mblt/Qwen3-VL-4B-Instruct_vision_transformer.mblt`
- Loads calibration data from: `./calibration_data/vision/npy_files.txt`
- **Loads rotation matrix** from: `./spinWeight/Qwen3-VL-4B-Instruct_text_model/R1/global_rotation.pth`
  - This matrix was generated during language model MXQ compilation
  - It ensures consistent quantization between vision and language components
- Applies advanced quantization with equivalent transformations:
  - **Head output channel rotation**: Aligns vision encoder outputs with language model inputs using the shared rotation matrix
- Configures 16-bit activations for the merger and deepstack merger `fc2` layers
- Uses multi-core compilation for vision encoder

**Key configurations (benchmark-best 4B encoder):**

- Calibration: mode 1 (MaxPercentile), output 0
- Activation 16-bit layers: `["model_merger_linear_fc2", "model_deepstack_merger_list_0_linear_fc2", "model_deepstack_merger_list_1_linear_fc2", "model_deepstack_merger_list_2_linear_fc2"]`
- Inference scheme: `all` for ARIES, `single` for REGULUS
- Equivalent transformations: QK, UD, VO, head_out_ch_rotation (using language model rotation matrix), SpinR2, optimize_ffn (SpinR1 disabled)
- Rotation matrix path: `./spinWeight/Qwen3-VL-4B-Instruct_text_model/R1/global_rotation.pth`

**Why the rotation matrix is needed:**
The vision encoder's output must be properly aligned with the language model's input space. The rotation matrix generated during language model quantization ensures that the vision features and text embeddings live in the same quantized space, maintaining accuracy when vision and language components are combined during inference.

**Output files:**

- `./mxq/Qwen3-VL-4B-Instruct_vision_transformer.mxq`: Quantized model ready for deployment

### Target device (`--target-device`)

Both MXQ compile scripts use `--target-device` to select the target NPU. REGULUS supports only `inference_scheme="single"`, which is selected automatically when a `regulus` device is specified. As in the ARIES flow, compile the language model first so the rotation matrix is available before compiling the vision encoder.

| User | `--target-device` |
|---|---|
| ARIES | `aries-rb` |
| REGULUS (customers from 2026-06) | `regulus-rb` |

> **Note:** VLM compilation is supported on newer REGULUS (`regulus-rb`, customers from 2026-06). Older REGULUS (`regulus-ra`, customers before 2026-06) do not support this workflow.

Outputs are written to the same `./mxq/` paths regardless of the target device.

### Step 3.3: Prepare Inference Configuration Files

After compiling both models to MXQ format, you need to prepare the configuration files for inference. This step downloads the necessary model configuration files and prepares them for use with the compiled MXQ models.

**Important:** This step must be done after completing both MXQ compilations (Steps 3.1 and 3.2) because it requires the rotation matrix from the language model compilation.

#### Get Model Configuration

First, download and prepare the model configuration file:

```bash
python get_config.py
```

**What it does:**

- Downloads `config.json` from the HuggingFace model repository
- Modifies the config to point to the compiled MXQ model files:
  - Sets `mxq_path` to `"Qwen3-VL-4B-Instruct_text_model.mxq"`
  - Sets `vision_config.mxq_path` to `"Qwen3-VL-4B-Instruct_vision_transformer.mxq"`
- Updates model architecture settings:
  - Changes `architectures` to `["MobilintQwen3VLForConditionalGeneration"]`
  - Changes `model_type` to `'mobilint-qwen3_vl'`
  - Sets `tie_word_embeddings` to `false` (the rotated embedding is provided separately)
- Saves the modified config to `./mxq/config.json`

#### Get Rotated Token Embedding Weight

Next, download and prepare the token embedding weight (`model.language_model.embed_tokens.weight`) with rotation:

```bash
python get_safetensors.py
```

**What it does:**

- Downloads `model.safetensors` from HuggingFace (Qwen3-VL-4B ships a single, unsharded safetensors file)
- Extracts the token embedding weight (`model.language_model.embed_tokens.weight`) — the lookup table that maps token IDs to hidden state vectors
- Applies the rotation matrix from the language model MXQ compilation:
  - Loads rotation matrix from: `./spinWeight/Qwen3-VL-4B-Instruct_text_model/R1/global_rotation.pth`
  - Right-multiplies the token embedding weight with the rotation matrix (`W @ R1`) to align with quantized space
- Saves the result to `./mxq/model.safetensors`

**Why token embedding rotation is needed:**
During MXQ compilation, the `SpinR1` equivalent transformation rotates the language model's internal weights into a transformed space. However, the token embedding layer is not included in MXQ compilation — it runs as a CPU lookup at inference time. Therefore, the token embedding weight must be pre-rotated with the same rotation matrix so that its output vectors match the quantized model's input space.

**Note:** The output filename `model.safetensors` is required by HuggingFace's `PreTrainedModel.from_pretrained()` convention. Despite the generic name, this file contains **only the rotated token embedding weight**, not the full model weights.

**Output files:**

- `./mxq/config.json`: Modified model configuration pointing to MXQ files
- `./mxq/model.safetensors`: Rotated token embedding weight (`model.language_model.embed_tokens.weight`)

**Important:** After running these scripts, you will have all 4 files needed for inference in the `./mxq/` directory:

1. `Qwen3-VL-4B-Instruct_text_model.mxq` (compiled language model)
2. `Qwen3-VL-4B-Instruct_vision_transformer.mxq` (compiled vision encoder)
3. `config.json` (model configuration)
4. `model.safetensors` (rotated token embedding weight)

No additional file copying is required.

## Complete Compilation Pipeline

Use the following command sequence to compile the full VLM:

```bash
# Stage 1: Calibration Data Generation

# Download calibration images from COCO dataset
python download_images.py

# Generate calibration data (language + vision)
python generate_calibration_data.py \
    --model-name Qwen/Qwen3-VL-4B-Instruct \
    --output-dir ./calibration_data \
    --num-samples 100 \
    --max-new-tokens 512

# Stage 2: MBLT Compilation

# Compile language model to MBLT (--target-device required)
python mblt_compile_language.py --target-device aries-rb

# Compile vision encoder to MBLT
python mblt_compile_vision.py --target-device aries-rb

# Stage 3: MXQ Compilation and Inference Preparation
# IMPORTANT: Compile language model FIRST (generates rotation matrix)
python mxq_compile_language.py --target-device aries-rb

# Then compile vision encoder (uses rotation matrix from language model)
python mxq_compile_vision.py --target-device aries-rb

# Prepare inference files (config.json and rotated token embedding)
python get_config.py
python get_safetensors.py

# All required files are now in the mxq/ directory:
# - Qwen3-VL-4B-Instruct_text_model.mxq
# - Qwen3-VL-4B-Instruct_vision_transformer.mxq
# - config.json
# - model.safetensors
```

## Understanding the Compilation Flow

### Language Model Pipeline

```text
[Download Images] -> images/*.jpg (100 COCO images)
    |
Original Model (HF) + Calibration Images
    |
[Calibration] -> calibration_data/language/*.npy
    |
[MBLT Compile] -> Qwen3-VL-4B-Instruct_text_model.mblt
    |
[MXQ Compile] -> Qwen3-VL-4B-Instruct_text_model.mxq
    |
    +-> global_rotation.pth (needed for vision encoder)
```

### Vision Encoder Pipeline

```text
[Download Images] -> images/*.jpg (100 COCO images)
    |
Original Model (HF) + Calibration Images
    |
[Calibration] -> calibration_data/vision/*.npy
    |
[MBLT Compile] -> Qwen3-VL-4B-Instruct_vision_transformer.mblt
    |
[MXQ Compile] -> Qwen3-VL-4B-Instruct_vision_transformer.mxq
    |            (Requires: global_rotation.pth from language model)
```

### Configuration Files Preparation

```text
[get_config.py] -> config.json
                   (Modified with MXQ paths)

[get_safetensors.py] -> model.safetensors
                        (Rotated token embedding weight)
```

### Key Dependencies

1. Vision encoder MXQ compilation **requires** the rotation matrix from language model MXQ compilation
2. Always run `mxq_compile_language.py` **before** `mxq_compile_vision.py`
3. Both MBLT files can be compiled independently, but MXQ files must follow the order above
4. `get_safetensors.py` requires the rotation matrix from language model MXQ compilation
5. All 4 output files (2 MXQ models, config.json, model.safetensors) must be in the same directory

## Output Summary

After completing all stages, you will have:

### Calibration Data

- `calibration_data/language/`: Language model calibration samples with metadata
- `calibration_data/vision/`: Vision encoder calibration samples with metadata

### MBLT Models (Hardware-Agnostic) - in `mblt/`

- `Qwen3-VL-4B-Instruct_text_model.mblt`: Language model in MBLT format
- `Qwen3-VL-4B-Instruct_vision_transformer.mblt`: Vision encoder in MBLT format

### MXQ Models and Deployment Files - in `mxq/`

All files needed for deployment are in this single directory:

- `Qwen3-VL-4B-Instruct_text_model.mxq`: Quantized language model
- `Qwen3-VL-4B-Instruct_vision_transformer.mxq`: Quantized vision encoder
- `config.json`: Model configuration with MXQ paths
- `model.safetensors`: Rotated token embedding weight (`model.language_model.embed_tokens.weight`)

## Troubleshooting

### Out of Memory (OOM) Errors

- Reduce `--num-samples` in calibration scripts
- Reduce `--max-new-tokens` in language calibration
- Close other GPU-intensive applications

### Missing Rotation Matrix Error

If vision encoder MXQ compilation fails with a missing rotation matrix error:

```bash
FileNotFoundError: ./spinWeight/Qwen3-VL-4B-Instruct_text_model/R1/global_rotation.pth
```

**Solution:** Run `mxq_compile_language.py` first to generate the rotation matrix.

### Calibration Data Not Found

Ensure the calibration data paths in the MXQ compile scripts match your actual calibration data location:

- Language: `./calibration_data/language/npy_files.json`
- Vision: Update the path in `mxq_compile_vision.py` if your data is elsewhere

### Model Download Issues

- Ensure you have accepted the model agreement on HuggingFace
- Verify your access token is valid: `huggingface-cli whoami`
- Check your internet connection and HuggingFace status

### No Images Found

```bash
FileNotFoundError: No images found in images/ directory
```

**Solution:** Run the image download script:

```bash
python download_images.py
```

This will download 100 images from COCO dataset to the `images/` directory.

## Deployment

After completing all compilation stages, the `./mxq/` directory contains all 4 files needed for deployment:

1. **Qwen3-VL-4B-Instruct_text_model.mxq** - Compiled language model
2. **Qwen3-VL-4B-Instruct_vision_transformer.mxq** - Compiled vision encoder
3. **config.json** - Model configuration with MXQ paths
4. **model.safetensors** - Rotated token embedding weight (`model.language_model.embed_tokens.weight`)

These files are ready for deployment on the NPU with the Mobilint runtime.

## Next Steps: Running Inference

To run inference with your compiled models, see the [Runtime Inference Tutorial](../../runtime/python/vlm/README.md).

The runtime tutorial demonstrates how to:

- Load compiled MXQ models using mblt-model-zoo
- Run image-text-to-text inference
- Customize prompts and generation parameters
- Handle multi-turn conversations
- Process multiple images

## References

- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [Mobilint Documentation](https://docs.mobilint.com)

## Support

For issues or questions:

- Check the troubleshooting section above
- Review qbcompiler SDK documentation
- Contact Mobilint support with detailed error logs

---

**Note:** This tutorial demonstrates the complete pipeline for VLM compilation. The techniques shown here can be adapted for other vision-language models with appropriate modifications to the model loading and patching code.
