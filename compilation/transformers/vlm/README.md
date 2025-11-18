# Vision Language Model (VLM) Compilation

This tutorial provides detailed instructions for compiling Vision Language Models (VLMs) using the Mobilint qubee compiler.

In this tutorial, we will use the [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) model, a state-of-the-art vision-language model developed by Qwen.

> ** Important Disclaimer:**
>
> The code in this tutorial requires **qubee version >= 0.12** with equivalent transformation support, which will be available in a near future release. The current version (v0.11.0.1) does not yet support the equivalent transformation features used in the MXQ compilation scripts, and compilation will fail with the current qubee version.
>

## Overview

The VLM compilation process consists of three main stages:

1. **Calibration Data Generation**: Create calibration datasets for quantization
2. **MBLT Compilation**: Compile the model to MBLT (Mobilint Binary Layout) format
3. **MXQ Compilation**: Apply advanced quantization and compile to MXQ format for deployment

The compilation process is performed separately for the **language model** (decoder) and **vision encoder** components.

After compilation, you will have all necessary files in the `mxq/` directory ready for deployment on Aries 2 hardware.

## Prerequisites

Before starting, ensure you have:

- Python 3.8 or higher
- qubee SDK installed (version >= 0.12 required)
- (optional) CUDA-capable GPU for calibration and compilation
- Sufficient disk space (~20GB for model + calibration data)

### Install Required Dependencies

Install the required Python packages for compilation:

```bash
pip install transformers==4.50.3 torch torchvision qwen-vl-utils datasets
```

### Download Calibration Images

The calibration process uses images from the COCO dataset. A download script is provided to automatically fetch 100 images:

```bash
cd calibration
python download_images.py
```

**What it does:**
- Downloads 100 images from the COCO 2017 validation set using HuggingFace datasets
- Automatically resizes images to 224x224 resolution
- Saves images to the `images/` directory as JPEG files
- If COCO download fails, generates synthetic sample images as fallback

**Output:**
- `images/image_0000.jpg` through `images/image_0099.jpg`

The calibration scripts will automatically use all images in the `images/` directory and cycle through diverse prompts (detailed descriptions, visual reasoning, counting, spatial understanding, etc.) to ensure calibration diversity.

## Stage 1: Calibration Data Generation

Calibration data is essential for quantization, as it helps the compiler understand the typical activation ranges of the model.

### Step 1.1: Generate Language Model Calibration Data

Generate calibration data for the language model (decoder):

```bash
cd calibration
python generate_language_calibration_data.py \
    --model-name Qwen/Qwen2-VL-2B-Instruct \
    --output-dir ./calibration_data/language \
    --num-samples 100 \
    --max-new-tokens 500
```

**Parameters:**
- `--model-name`: HuggingFace model identifier
- `--output-dir`: Directory to save calibration data
- `--num-samples`: Number of calibration samples (default: all available images)
- `--max-new-tokens`: Maximum tokens to generate per sample (captures longer sequences)

**What it does:**
- Loads all images from `images/` folder (100 JPEG images downloaded earlier)
- Cycles through 20 diverse prompt types (object identification, detailed description, visual reasoning, spatial understanding, etc.)
- Captures `inputs_embeds` tensors after vision features are merged into text embeddings
- Saves calibration data as `.npy` files with metadata

**Output structure:**
```
calibration_data/language/
 sample_000/
    inputs_embeds.npy    # [1, seq_len, 1536]
 sample_001/
    inputs_embeds.npy
 ...
 metadata.json            # Sample information
 npy_files.txt           # List of all .npy paths
```

### Step 1.2: Generate Vision Encoder Calibration Data

Generate calibration data for the vision encoder:

```bash
python generate_vision_calibration_data.py \
    --model-name Qwen/Qwen2-VL-2B-Instruct \
    --output-dir ./calibration_data/vision \
    --num-samples 100
```

**Parameters:**
- `--model-name`: HuggingFace model identifier
- `--output-dir`: Directory to save calibration data
- `--num-samples`: Number of calibration samples (default: all available images)

**Note:** Image size is fixed at 224x224 for vision encoder calibration.

**What it does:**
- Loads all images from `images/` folder (100 JPEG images downloaded earlier)
- Cycles through diverse prompts (same as language calibration)
- Captures vision encoder inputs (pixel values)
- Reshapes to format compatible with Aries 2 architecture: `[896, 56, 6]`
- Saves calibration data as `.npy` files with metadata

**Output structure:**
```
calibration_data/vision/
 sample_000/
    images.npy          # [896, 56, 6]
 sample_001/
    images.npy
 ...
 metadata.json           # Sample information
 npy_files.txt          # List of all .npy paths
```

## Stage 2: MBLT Compilation

MBLT (Mobilint Binary Layout) is an intermediate format that represents the model graph and weights in a hardware-agnostic way.

### Step 2.1: Compile Language Model to MBLT

Compile the language model (decoder) to MBLT format:

```bash
cd ../compile
python mblt_compile_language.py
```

**What it does:**
- Captures language model inputs during sample generation
- Marks sequence length dimensions as dynamic (for variable-length inputs)
- Applies Aries 2-compatible architectural patches:
  - **Pre-cached RoPE embeddings**: Eliminates runtime trigonometric operations
  - **Last-query slicing**: Optimizes the final decoder layer for decode phase
  - **Stateful KV cache wrappers**: Enables efficient auto-regressive generation
  - **Dynamic shape handling**: Supports variable sequence lengths
- Compiles the model using qubee's ModelParser with PyTorch FX tracing
- Configures dynamic shapes for attention operators
- Serializes to MBLT binary format
- Validates output by comparing with original model

**Key transformations:**
- Input embeddings dimension marked as dynamic: `[batch, seq_len, hidden_size]`
- Attention mask and position IDs marked as dynamic for variable sequences
- Cache position marked as dynamic for auto-regressive generation
- RoPE embeddings pre-computed for max sequence length (16384)

**Output files:**
- `qwen2vl_language.mblt`: Compiled model in MBLT format
- `qwen2vl_language.infer`: Inference values for validation
- `qwen2vl_language.json`: Comparison results with original model

### Step 2.2: Compile Vision Encoder to MBLT

Compile the vision encoder to MBLT format:

```bash
python mblt_compile_vision.py
```

**What it does:**
- Captures vision encoder inputs during sample inference
- Reprocesses pixel values to Aries 2-compatible format
- Applies Aries 2-compatible architectural patches:
  - **3D2D convolution**: Transforms 3D convolutions to 2D for NPU optimization
  - **Split QKV projection**: Separates Query, Key, Value projections for better parallelization
  - **Pre-computed RoPE embeddings**: Eliminates runtime trigonometric operations
  - **Merged patchify operation**: Reduces memory transfers
- Compiles the model using qubee's ModelParser with PyTorch FX tracing
- Sets data format to NHWC (Aries 2-friendly) for all input constants
- Serializes to MBLT binary format
- Validates output by comparing with original model

**Key transformations:**
- Pixel values reprocessed from HuggingFace format `[num_patches, channels*patch_size^2]` to Aries 2 format `[batch, channels*temporal, height, width]`
- 3D temporal convolutions converted to 2D spatial convolutions
- QKV attention projections split for parallel execution
- RoPE embeddings pre-computed based on image grid dimensions

**Output files:**
- `qwen2vl_vision.mblt`: Compiled model in MBLT format
- `qwen2vl_vision.infer`: Inference values for validation
- `qwen2vl_vision.json`: Comparison results with original model

## Stage 3: MXQ Compilation (Advanced Quantization)

MXQ (Mobilint Quantized) format applies advanced quantization techniques and prepares the model for deployment on Aries 2 hardware.

### Step 3.1: Compile Language Model to MXQ

Compile the language model from MBLT to MXQ format:

```bash
python mxq_compile_language.py
```

**What it does:**
- Loads the MBLT file: `qwen2vl_language.mblt`
- Loads calibration data from: `../calibration/calibration_data/language/npy_files.txt`
- Applies advanced quantization with equivalent transformations.
- Configures 16-bit activations for input embeddings: `inputs_embeds/reshape`
- Uses single-core compilation for language model
- Enables LLM-specific optimizations
- **Generates rotation matrix** at: `/tmp/qubee/spinWeight/qwen2vl_language/R1/global_rotation.pth`
  - This rotation matrix is **required for vision encoder MXQ compilation**

**Key configurations:**
- Calibration mode: 1 (standard calibration)
- Activation 16-bit layers: `["inputs_embeds/reshape"]`
- Inference scheme: `single` (single-core execution)
- Single-core compile: `True` (optimized for language models)
- Equivalent transformations: QK, UD (with learning), SPIN R1, SPIN R2

**Output files:**
- `./mxq/Qwen2-VL-2B-Instruct_text_model.mxq`: Quantized model ready for Aries 2 deployment
- `/tmp/qubee/spinWeight/qwen2vl_language/R1/global_rotation.pth`: Global rotation matrix (needed for vision encoder)

### Step 3.2: Compile Vision Encoder to MXQ

**Important:** You must complete Step 3.1 (language model MXQ compilation) first, as the vision encoder compilation requires the rotation matrix generated during language model compilation.

Compile the vision encoder from MBLT to MXQ format:

```bash
python mxq_compile_vision.py
```

**What it does:**
- Loads the MBLT file: `qwen2vl_vision.mblt`
- Loads calibration data from: `/workspace/data_prep/calibration_data/vision/npy_files.txt`
- **Loads rotation matrix** from: `/tmp/qubee/spinWeight/qwen2vl_language/R1/global_rotation.pth`
  - This matrix was generated during language model MXQ compilation
  - It ensures consistent quantization between vision and language components
- Applies advanced quantization with equivalent transformations:
  - **Head output channel rotation**: Aligns vision encoder outputs with language model inputs using the shared rotation matrix
- Configures 16-bit activations for merger layer: `model_merger_fc2` (This may not be necessary in the future)
- Uses multi-core compilation for vision encoder

**Key configurations:**
- Calibration output mode: 1 (standard output calibration)
- Activation 16-bit layers: `["model_merger_fc2"]`
- Inference scheme: `multi` (multi-core execution)
- Equivalent transformations: Head output channel rotation (using language model rotation matrix)
- Rotation matrix path: `/tmp/qubee/spinWeight/qwen2vl_language/R1/global_rotation.pth`

**Why the rotation matrix is needed:**
The vision encoder's output must be properly aligned with the language model's input space. The rotation matrix generated during language model quantization ensures that the vision features and text embeddings live in the same quantized space, maintaining accuracy when vision and language components are combined during inference.

**Output files:**
- `./mxq/Qwen2-VL-2B-Instruct_vision_transformer.mxq`: Quantized model ready for Aries 2 deployment

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
- Creates a `./mxq/` directory for inference files
- Modifies the config to point to the compiled MXQ model files:
  - Sets `mxq_path` to `"Qwen2-VL-2B-Instruct_text_model.mxq"`
  - Sets `vision_config.mxq_path` to `"Qwen2-VL-2B-Instruct_vision_transformer.mxq"`
- Updates model architecture settings:
  - Changes `architectures` to `["MobilintQwen2VLForConditionalGeneration"]`
  - Changes `model_type` to `'mobilint-qwen2_vl'`
  - Sets `max_position_embeddings` to 32768
  - Sets `sliding_window` to 32768
  - Enables `tie_word_embeddings`
- Saves the modified config to `./mxq/config.json`

#### Get Model Embeddings

Next, download and prepare the embedding weights with proper rotation:

```bash
python get_safetensors.py
```

**What it does:**
- Downloads `model-00001-of-00002.safetensors` from HuggingFace (contains embedding weights)
- Extracts the `model.embed_tokens.weight` tensor
- Applies the rotation matrix from the language model MXQ compilation:
  - Loads rotation matrix from: `/tmp/qubee/spinWeight/qwen2vl_language/R1/global_rotation.pth`
  - Multiplies the embedding tensor with the rotation matrix to align with quantized space
- Saves the rotated embedding tensor to `./mxq/model.safetensors`

**Why embedding rotation is needed:**
The embedding layer needs to be rotated with the same rotation matrix used during language model quantization. This ensures that the input embeddings are in the same quantized space as the rest of the language model, maintaining accuracy and consistency throughout the inference pipeline.

**Output files:**
- `./mxq/config.json`: Modified model configuration pointing to MXQ files
- `./mxq/model.safetensors`: Rotated embedding weights aligned with quantized model

**Important:** After running these scripts, you will have all 4 files needed for inference in the `./mxq/` directory:
1. `Qwen2-VL-2B-Instruct_text_model.mxq` (compiled language model)
2. `Qwen2-VL-2B-Instruct_vision_transformer.mxq` (compiled vision encoder)
3. `config.json` (model configuration)
4. `model.safetensors` (rotated embeddings)

No additional file copying is required!

## Complete Compilation Pipeline

Here's the complete sequence of commands to compile the full VLM:

```bash
# Stage 1: Calibration Data Generation
cd /workspace/mblt-sdk-tutorial/compilation/vlm/calibration

# Download calibration images from COCO dataset
python download_images.py

# Generate language calibration data
python generate_language_calibration_data.py \
    --model-name Qwen/Qwen2-VL-2B-Instruct \
    --output-dir ./calibration_data/language \
    --num-samples 100 \
    --max-new-tokens 500

# Generate vision calibration data
python generate_vision_calibration_data.py \
    --model-name Qwen/Qwen2-VL-2B-Instruct \
    --output-dir ./calibration_data/vision \
    --num-samples 100

# Stage 2: MBLT Compilation
cd ../compile

# Compile language model to MBLT
python mblt_compile_language.py

# Compile vision encoder to MBLT
python mblt_compile_vision.py

# Stage 3: MXQ Compilation and Inference Preparation
# IMPORTANT: Compile language model FIRST (generates rotation matrix)
python mxq_compile_language.py

# Then compile vision encoder (uses rotation matrix from language model)
python mxq_compile_vision.py

# Prepare inference configuration files (config.json and model.safetensors)
python get_config.py
python get_safetensors.py

# All required files are now in the mxq/ directory:
# - Qwen2-VL-2B-Instruct_text_model.mxq
# - Qwen2-VL-2B-Instruct_vision_transformer.mxq
# - config.json
# - model.safetensors
```

## Understanding the Compilation Flow

### Language Model Pipeline
```
[Download Images] -> images/*.jpg (100 COCO images)
    |
Original Model (HF) + Calibration Images
    |
[Calibration] -> calibration_data/language/*.npy
    |
[MBLT Compile] -> Qwen2-VL-2B-Instruct_text_model.mblt
    |
[MXQ Compile] -> Qwen2-VL-2B-Instruct_text_model.mxq
    |
    +-> global_rotation.pth (needed for vision encoder)
```

### Vision Encoder Pipeline
```
[Download Images] -> images/*.jpg (100 COCO images)
    |
Original Model (HF) + Calibration Images
    |
[Calibration] -> calibration_data/vision/*.npy
    |
[MBLT Compile] -> Qwen2-VL-2B-Instruct_vision_transformer.mblt
    |
[MXQ Compile] -> Qwen2-VL-2B-Instruct_vision_transformer.mxq
    |            (Requires: global_rotation.pth from language model)
```

### Configuration Files Preparation
```
[get_config.py] -> config.json
                   (Modified with MXQ paths)

[get_safetensors.py] -> model.safetensors
                        (Embedding weights with rotation applied)
```

### Key Dependencies
1. Vision encoder MXQ compilation **requires** the rotation matrix from language model MXQ compilation
2. Always run `mxq_compile_language.py` **before** `mxq_compile_vision.py`
3. Both MBLT files can be compiled independently, but MXQ files must follow the order above
4. `get_safetensors.py` requires the rotation matrix from language model MXQ compilation
5. All 4 output files (2 MXQ models, config.json, model.safetensors) must be in the same directory for deployment

## Output Summary

After completing all stages, you will have:

### Calibration Data
- `calibration_data/language/`: Language model calibration samples with metadata
- `calibration_data/vision/`: Vision encoder calibration samples with metadata

### MBLT Models (Hardware-Agnostic) - in `compile/mblt/`
- `Qwen2-VL-2B-Instruct_text_model.mblt`: Language model in MBLT format
- `Qwen2-VL-2B-Instruct_vision_transformer.mblt`: Vision encoder in MBLT format

### MXQ Models and Deployment Files - in `compile/mxq/`
All files needed for deployment are in this single directory:
- `Qwen2-VL-2B-Instruct_text_model.mxq`: Quantized language model
- `Qwen2-VL-2B-Instruct_vision_transformer.mxq`: Quantized vision encoder
- `config.json`: Model configuration with MXQ paths
- `model.safetensors`: Rotated embedding weights

### Validation Files - in `compile/`
- `*.infer`: Inference values for validation
- `*.json`: Comparison results with original models

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `--num-samples` in calibration scripts
- Reduce `--max-new-tokens` in language calibration
- Close other GPU-intensive applications

### Missing Rotation Matrix Error
If vision encoder MXQ compilation fails with a missing rotation matrix error:
```
FileNotFoundError: /tmp/qubee/spinWeight/qwen2vl_language/R1/global_rotation.pth
```
**Solution:** Run `mxq_compile_language.py` first to generate the rotation matrix.

### Calibration Data Not Found
Ensure the calibration data paths in the MXQ compile scripts match your actual calibration data location:
- Language: `../calibration/calibration_data/language/npy_files.txt`
- Vision: Update the path in `mxq_compile_vision.py` if your data is elsewhere

### Model Download Issues
- Ensure you have accepted the model agreement on HuggingFace
- Verify your access token is valid: `huggingface-cli whoami`
- Check your internet connection and HuggingFace status

### No Images Found
```
FileNotFoundError: No images found in images/ directory
```
**Solution:** Run the image download script:
```bash
cd calibration
python download_images.py
```
This will download 100 images from COCO dataset to the `images/` directory.

## Deployment

After completing all compilation stages, the `./mxq/` directory contains all 4 files needed for deployment:

1. **Qwen2-VL-2B-Instruct_text_model.mxq** - Compiled language model
2. **Qwen2-VL-2B-Instruct_vision_transformer.mxq** - Compiled vision encoder
3. **config.json** - Model configuration with MXQ paths
4. **model.safetensors** - Rotated embedding weights

These files are ready for deployment on Aries 2 hardware using the Mobilint runtime.

## Next Steps: Running Inference

To run inference with your compiled models, see the [Runtime Inference Tutorial](../../../runtime/transformers/vlm/README.md).

The runtime tutorial demonstrates how to:
- Load compiled MXQ models using mblt-model-zoo
- Run image-text-to-text inference
- Customize prompts and generation parameters
- Handle multi-turn conversations
- Process multiple images

## References

- [Qwen2-VL Model Card](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [Mobilint Documentation](https://docs.mobilint.com)
- [qubee SDK Documentation](https://docs.mobilint.com/qubee)
- [Aries 2 Architecture Guide](https://docs.mobilint.com/aries2)

## Support

For issues or questions:
- Check the troubleshooting section above
- Review qubee SDK documentation
- Contact Mobilint support with detailed error logs

---

**Note:** This tutorial demonstrates the complete pipeline for VLM compilation. The techniques shown here can be adapted for other vision-language models with appropriate modifications to the model loading and patching code.
