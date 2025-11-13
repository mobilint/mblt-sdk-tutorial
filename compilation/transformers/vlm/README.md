# Vision Language Model (VLM) Compilation

This tutorial provides detailed instructions for compiling Vision Language Models (VLMs) using the Mobilint qubee compiler.

In this tutorial, we will use the [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) model, a state-of-the-art vision-language model developed by Qwen.

> ** Important Disclaimer:**
>
> The code in this tutorial requires **qubee version >= 0.12** with equivalent transformation support, which will be available in a near future release. The current version (v0.11.0.1) does not yet support the equivalent transformation features used in the MXQ compilation scripts, and compilation will fail with the current qubee version.
>

## Overview

The VLM compilation and deployment process consists of four main stages:

1. **Calibration Data Generation**: Create calibration datasets for quantization
2. **MBLT Compilation**: Compile the model to MBLT (Mobilint Binary Layout) format
3. **MXQ Compilation**: Apply advanced quantization and compile to MXQ format for deployment
4. **Inference**: Run the compiled model on Aries 2 hardware using the Mobilint runtime

The compilation process is performed separately for the **language model** (decoder) and **vision encoder** components.

## Prerequisites

Before starting, ensure you have:

- Python 3.8 or higher
- qubee SDK installed (version >= 0.12 required)
- CUDA-capable GPU for calibration and compilation
- HuggingFace account with access to the Qwen2-VL model
- Sufficient disk space (~10GB for model + calibration data)

### Model Preparation

First, prepare the model by signing up on [HuggingFace](https://huggingface.co/) and signing the agreement on the [Qwen2-VL-2B-Instruct model page](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct).

Then, obtain an access token from the [HuggingFace token management page](https://huggingface.co/settings/tokens). A read-only token is sufficient for downloading the model.

Set up authentication via the HuggingFace CLI:

```bash
pip install huggingface_hub[cli]
huggingface-cli login
# Enter your access token when prompted
```

Download the model:

```bash
apt-get install git-lfs  # Install git-lfs if not installed
git lfs install          # Initialize git-lfs
git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
```

### Install Required Dependencies

Install the required Python packages for compilation:

```bash
pip install transformers==4.50.3 torch torchvision qwen-vl-utils datasets
```

For inference, you'll also need:

```bash
pip install mblt-model-zoo
```

### Prepare Calibration Images

The calibration scripts expect images in a `converted_pngs/` directory. Create this directory and add your calibration images (PNG format):

```bash
mkdir -p converted_pngs
# Add your PNG images to this directory
# The scripts will use all images in this folder for calibration
```

The calibration scripts will automatically cycle through diverse prompts (detailed descriptions, visual reasoning, counting, spatial understanding, etc.) to ensure calibration diversity across all your images.

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
- Loads all images from `converted_pngs/` folder
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
- Loads all images from `converted_pngs/` folder
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
- `qwen2vl_language.mxq`: Quantized model ready for Aries 2 deployment
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
- `qwen2vl_vision.mxq`: Quantized model ready for Aries 2 deployment

## Complete Compilation and Inference Pipeline

Here's the complete sequence of commands to compile and run the full VLM:

```bash
# Stage 1: Calibration Data Generation
cd /workspace/mblt-sdk-tutorial/compilation/vlm/calibration

# Generate language calibration data
python generate_language_calibration_data.py \
    --model-name Qwen/Qwen2-VL-2B-Instruct \
    --output-dir ./calibration_data/language \
    --image-size 224 224 \
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

# Stage 3: MXQ Compilation
# IMPORTANT: Compile language model FIRST (generates rotation matrix)
python mxq_compile_language.py

# Then compile vision encoder (uses rotation matrix from language model)
python mxq_compile_vision.py

# Stage 4: Run Inference
cd ../inference
python run_qwen2_vl_local.py
```

## Understanding the Compilation Flow

### Language Model Pipeline
```
Original Model (HF)
    
[Calibration]  calibration_data/language/*.npy
    
[MBLT Compile]  qwen2vl_language.mblt
    
[MXQ Compile]  qwen2vl_language.mxq
     global_rotation.pth (needed for vision encoder)
```

### Vision Encoder Pipeline
```
Original Model (HF)
    
[Calibration]  calibration_data/vision/*.npy
    
[MBLT Compile]  qwen2vl_vision.mblt
    
[MXQ Compile]  qwen2vl_vision.mxq
    
     Requires: global_rotation.pth (from language model)
```

### Combined Inference Flow
```
Image Input
    
[Vision Encoder MXQ]  Vision features
    
[Merge with text embeddings]
    
[Language Model MXQ]  Generated text output
```

### Key Dependencies
1. Vision encoder MXQ compilation **requires** the rotation matrix from language model MXQ compilation
2. Always run `mxq_compile_language.py` **before** `mxq_compile_vision.py`
3. Both MBLT files can be compiled independently, but MXQ files must follow the order above
4. Inference requires both compiled MXQ models and the model configuration files

## Output Summary

After completing all stages, you will have:

### Calibration Data
- `calibration_data/language/`: Language model calibration samples with metadata
- `calibration_data/vision/`: Vision encoder calibration samples with metadata

### MBLT Models (Hardware-Agnostic)
- `qwen2vl_language.mblt`: Language model in MBLT format
- `qwen2vl_vision.mblt`: Vision encoder in MBLT format

### MXQ Models (Aries 2-Optimized)
- `qwen2vl_language.mxq`: Quantized language model ready for deployment
- `qwen2vl_vision.mxq`: Quantized vision encoder ready for deployment

### Validation Files
- `*.infer`: Inference values for validation
- `*.json`: Comparison results with original models

### Inference Files
- `inference/run_qwen2_vl_local.py`: Example inference script
- `inference/*.json`: Model configuration files for inference
- `inference/*.mxq`: Compiled models (copy from compile/ directory)

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

### No Images in converted_pngs/
```
FileNotFoundError: No PNG images found in converted_pngs
```
**Solution:** Create the directory and add PNG images:
```bash
mkdir -p converted_pngs
# Copy your calibration images (PNG format) to this directory
```

## Stage 4: Running Inference

Once you have compiled the model to MXQ format, you can run inference using the Mobilint runtime.

### Prerequisites

Before running inference, ensure you have:
- Completed all compilation stages (calibration, MBLT, and MXQ)
- The compiled MXQ files: `qwen2vl_language.mxq` and `qwen2vl_vision.mxq`
- Model configuration files (tokenizer, processor config, etc.)
- `mblt-model-zoo` package installed

### Setting Up the Inference Directory

The `inference/` directory should contain:
- **Compiled models**: `qwen2vl_language.mxq` and `qwen2vl_vision.mxq`
- **Configuration files**: All the JSON and model configuration files from the original HuggingFace model
  - `config.json` - Model architecture configuration
  - `tokenizer.json` - Tokenizer vocabulary
  - `tokenizer_config.json` - Tokenizer settings
  - `preprocessor_config.json` - Vision preprocessor config
  - `video_preprocessor_config.json` - Video processing config
  - `generation_config.json` - Text generation parameters
  - `chat_template.json` - Chat format template
  - `vocab.json` - Vocabulary mapping
  - `model.safetensors` - Original model weights (for reference)
- **Inference script**: `run_qwen2_vl_local.py`

### Running Inference

Navigate to the inference directory and run the script:

```bash
cd inference
python run_qwen2_vl_local.py
```

### Understanding the Inference Code

The inference script (`run_qwen2_vl_local.py`) demonstrates how to:

```python
from transformers import TextStreamer
from mblt_model_zoo.transformers import pipeline, AutoProcessor
from PIL import Image

# Load processor
processor = AutoProcessor.from_pretrained(".", use_fast=True)

# Create pipeline
pipe = pipeline(
    "image-text-to-text",
    model=".",  # Current directory containing config files
    processor=processor,
)

# Prepare messages with image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.jpg"},
            {"type": "text", "text": "Your question here"},
        ],
    }
]

# Run inference with streaming
pipe(
    text=messages,
    generate_kwargs={
        "max_length": 512,
        "streamer": TextStreamer(tokenizer=pipe.tokenizer, skip_prompt=False),
    },
)

# Clean up
pipe.model.dispose()
```

### Key Features

- **Automatic Model Loading**: The pipeline automatically detects and loads the MXQ models
- **Image Input**: Supports both local file paths and URLs for images
- **Streaming Output**: Uses `TextStreamer` to display generated text in real-time
- **Multi-turn Conversations**: Can handle multi-turn dialogues by extending the messages list
- **Flexible Generation**: Configurable generation parameters like `max_length`

### Customizing Your Inference

You can modify the script to:

1. **Use local images**:
```python
{"type": "image", "image": "/path/to/your/image.jpg"}
```

2. **Change the prompt**:
```python
{"type": "text", "text": "What objects are in this image?"}
```

3. **Adjust generation parameters**:
```python
generate_kwargs={
    "max_length": 1024,  # Longer responses
    "temperature": 0.7,   # Creativity control
    "top_p": 0.9,        # Nucleus sampling
}
```

4. **Multi-turn conversation**:
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "image.jpg"},
            {"type": "text", "text": "What's in this image?"},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "I see a dog and a person."}],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "What color is the dog?"}],
    },
]
```

### Performance Notes

- The compiled MXQ models run on Aries 2 hardware for optimal performance
- The pipeline automatically manages the interaction between vision and language components
- Memory is efficiently managed through the stateful KV cache system
- Use `pipe.model.dispose()` to properly clean up resources after inference

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
