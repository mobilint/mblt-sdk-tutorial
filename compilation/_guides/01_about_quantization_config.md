# Compile Config Guide

This document introduces the config objects passed to `mxq_compile()`.

Quantization configs have diverse parameter combinations, and optimal values differ by model architecture and task.
Even identical settings can affect accuracy and performance differently depending on the model,
so it is recommended to reference compile scripts from similar models or tasks
in the tutorial directories (`image_classification/`, `llm/`, `vlm/`, etc.) as baseline configs.

---

## Config Types Overview

| Config | Role | Target |
|--------|------|--------|
| `CalibrationConfig` | Quantization range determination (per-channel, percentile, etc.) | All models requiring quantization |
| `BitConfig` | Quantization bit count per transformer component (8bit/4bit) | Transformer-based models (LLM, etc.) |
| `LlmConfig` | Sequence length, KV cache, NPU core allocation | Transformer decoder architecture (autoregressive + KV cache) |
| `EquivalentTransformationConfig` | Advanced mathematical transformations to reduce quantization error (SpinQuant, etc.) | Recommended for 4bit quantization |
| `SearchWeightScaleConfig` | Per-layer weight scale learning for quantization accuracy correction | Recommended for 4bit quantization |
| `PreprocessingConfig` | Automatic image preprocessing by compiler during calibration | Image input models (Image Classification, etc.) |

---

## CalibrationConfig

Configures how activation value ranges are determined during quantization.
Controls per-channel/per-tensor methods and percentile-based clipping.

> **Difference from PyTorch quantization**: PyTorch's default quantization uses min-max method,
> but qbcompiler uses percentile-based clipping to reduce the impact of outliers.

```python
from qbcompiler import CalibrationConfig

calibration_config = CalibrationConfig(
    # Select the combination of Weight quantization + Activation quantization methods.
    #   0: WChALayer     — Weight: per-channel, Activation: per-layer.
    #                       Channel-wise scale for weights, layer-wise scale for activations.
    #   1: WChAMulti     — Weight: per-channel, Activation: multi-layer.
    #                       Channel-wise for weights, activations use statistics aggregated
    #                       across multiple layers. More stable than single-layer quantization.
    #   2: WChALayerZeropoint  — Same as 0 + zeropoint applied to activations.
    #   3: WChAMultiZeropoint  — Same as 1 + zeropoint applied to activations.
    #                            Beneficial for asymmetric activation distributions.
    method=1,

    # Output quantization method.
    #   0: Layer   — Single scale applied to the entire layer output.
    #   1: Ch      — Individual scale per output channel.
    #   2: Sigmoid — Sigmoid-based quantization.
    output=0,

    # Clipping mode for determining quantization range.
    #   0: Max           — Use min/max activation values directly as quantization range.
    #                       Simple but sensitive to outliers.
    #   1: MaxPercentile — Determine clipping range based on percentile.
    #                       Reduces quantization error by excluding outliers.
    #   2: Histogram     — Search for optimal clipping range based on activation histogram.
    #                       Minimizes information loss using KL-divergence, etc.
    mode=1,

    # MaxPercentile detailed settings
    max_percentile=CalibrationConfig.MaxPercentile(
        # Specifies at which percentile to clip activation values.
        # 0.9999 = 99.99th percentile. Clips the top 0.01% outliers to prevent
        # the quantization range from being excessively expanded by outliers.
        percentile=0.9999,

        # Specifies what percentage of top values to preserve separately during percentile calculation.
        # 0.01 = top 1%. Manages important large activation values separately
        # to prevent accuracy loss from clipping.
        topk_ratio=0.01,
    ),
)
```

**Usage examples**:
- `image_classification/model_compile.py`
- `llm/generate_mxq.py`
- `bert/compile_mxq.py`

---

## BitConfig

Specifies the quantization bit count for each component of transformer layers.
Supports selecting 8bit or 4bit, or mixed settings such as keeping only value at 8bit.

```python
from qbcompiler import BitConfig

bit_config = BitConfig(
    transformer=BitConfig.Transformer(
        weight=BitConfig.Transformer.Weight(
            # Individually specify bit count for each attention layer projection weight.
            # All components can be set to the same bit count, or
            # mixed settings (e.g., w4v8) keeping some layers at 8bit are also possible.

            query=8,    # Q projection — weights that transform input to query vectors
            key=8,      # K projection — weights that transform input to key vectors
            value=8,    # V projection — weights that transform input to value vectors
            output=8,   # Output projection — weights that pass attention output to the next layer
            ffn=8,      # Feed-Forward Network — weights of the transformer's FFN block
            head=8,     # Attention head — weights of multi-head attention heads
        ),
    )
)
```

**Usage examples**:
- `llm/generate_mxq.py` - 8bit
- `llm/generate_mxq_4bit.py` - 4bit (w4, w4v8)

---

## LlmConfig

Configures sequence length, KV cache, and NPU core allocation for LLM compilation.
This is an LLM-specific config, but it is also used with sub-models that have autoregressive structure
like STT decoder, since they require the same KV cache management.

```python
from qbcompiler import LlmConfig

llm_config = LlmConfig(
    # Whether to enable LLM-specific settings
    apply=True,

    attributes=LlmConfig.Attributes(
        # Maximum number of tokens that can be input to the model at once.
        # Upper limit for prompt length processed during the prefill stage.
        max_data_length=4096,

        # Maximum length of the entire sequence including generation.
        # The sum of prefill input + generated tokens cannot exceed this value.
        max_sequence_length=4096,

        # Maximum number of tokens that can be stored in the KV cache.
        # Buffer size for caching previous tokens' key/value during autoregressive generation.
        # Larger values allow maintaining longer context but increase memory usage.
        max_cache_length=4096,

        # Data buffer size processed by a single NPU core at a time.
        # Hardware-level parameter that affects NPU internal memory allocation.
        max_core_data_length=128,

        calibration=LlmConfig.Attributes.Calibration(
            # True: Collect activation distributions using the full sequence length during calibration.
            # False: Use partial sequences. True is better for accuracy but uses more memory.
            use_full_seq_length=True,
        ),
    ),
)
```

**Usage examples**:
- `llm/generate_mxq.py` - Sequence/cache length settings for LLM compilation
- `stt/compile_decoder.py` - Whisper decoder (requires LlmConfig due to autoregressive structure)

---

## EquivalentTransformationConfig

Configures advanced mathematical transformations (SpinQuant, etc.) to reduce quantization accuracy loss.
Generates rotation matrices (spinWeight) during compilation.
Recommended for 4bit quantization where quantization error is large.

> **Note**: When using SpinR1, additional manual work is required to apply R1 rotation to embedding weights
> after compilation. See [SpinQuant (R1/R2) Details](#spinquant-r1r2-details) below.

```python
from qbcompiler import EquivalentTransformationConfig

et_config = EquivalentTransformationConfig(
    # Normalization-Convolution equivalent transformation.
    # Absorbs LayerNorm/RMSNorm scales into subsequent linear layers
    # to create quantization-friendly weight distributions.
    norm_conv=EquivalentTransformationConfig.NormConv(apply=True),

    # Query-Key equivalent transformation.
    # Rotates Q and K weights to reduce quantization error in attention scores.
    qk=EquivalentTransformationConfig.Qk(apply=True),

    # Up-Down equivalent transformation.
    # Rotates FFN up/down projection weights to reduce quantization error.
    ud=EquivalentTransformationConfig.Ud(apply=True),

    # Value-Output equivalent transformation.
    # Rotates V and output projection weights to reduce quantization error in attention output.
    vo=EquivalentTransformationConfig.Vo(apply=True),

    # SpinQuant R1 — Global rotation matrix applied to the entire model.
    # Rotates the weight space into a quantization-favorable distribution.
    # Generates spinWeight/{model}/R1/global_rotation.pth during compilation.
    # Embedding weights must have this rotation pre-applied for 4bit quantization.
    spin_r1=EquivalentTransformationConfig.SpinR1(apply=True),

    # SpinQuant R2 — Per-layer rotation matrices applied to each transformer layer.
    # Compensates for weight distribution differences across layers for finer optimization than R1.
    # Generates per-layer files in spinWeight/{model}/R2/ during compilation.
    spin_r2=EquivalentTransformationConfig.SpinR2(apply=True),

    # QK Rotation — Rotates Q/K weights while maintaining compatibility
    # with positional encodings like RoPE.
    qk_rotation=EquivalentTransformationConfig.QkRotation(apply=True),

    # FFN Multi-LUT — Decomposes FFN weights into multiple lookup tables
    # to increase representational power in 4bit quantization.
    feed_forward_multi_lut=EquivalentTransformationConfig.FeedForwardMultiLut(apply=True),

    # FFN optimization — Rearranges FFN block operation structure to fit the NPU.
    optimize_ffn=EquivalentTransformationConfig.OptimizeFfn(apply=True),
)
```

**Usage examples**:
- `llm/generate_mxq_4bit.py` - LLM 4bit SpinQuant application
- `vlm/mxq_compile_language.py` - VLM language model equivalent transformation
- `vlm/mxq_compile_vision.py` - VLM vision encoder R1 rotation matrix reference (`HeadOutChRotation`)

### SpinQuant (R1/R2) Details

SpinQuant is a technique that rotates the weight space to reduce accuracy loss in 4bit quantization.
(See [SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406))

Rotation matrix files are generated in the `spinWeight/` directory during compilation.

```text
spinWeight/{model_name}/
├── R1/
│   └── global_rotation.pth     # Global rotation matrix (one per model)
└── R2/
    └── layer_*.pth             # Per-layer rotation matrices (one per layer)
```

**R1 (Global Rotation)** transforms the entire model's weight space with a single rotation matrix.
This rotation is already reflected inside the compiled MXQ model, but since
**the embedding layer is not included in MXQ and runs on CPU**,
the same R1 rotation must be manually applied to embedding weights before inference.

- When not using SpinQuant(R1): Embedding rotation not needed
- When using SpinQuant(R1): R1 rotation on embeddings is required

LLM embedding rotation example (`llm/get_rotation_emb.py`):

```python
# Load original embedding weights [vocab_size, embed_dim]
emb = torch.load("embedding.pt")

# Load R1 rotation matrix generated during compilation
rot = torch.jit.load("spinWeight/model/R1/global_rotation.pth")
rot_matrix = next(rot.parameters())

# Apply R1 rotation to embeddings (maintain precision in float64, then convert to bfloat16)
emb = (emb.double() @ rot_matrix.double()).bfloat16()
torch.save(emb, "embedding_rot.pt")
```

VLM text embedding rotation example (`vlm/get_safetensors.py`):

```python
# Extract text embedding from HuggingFace safetensors
with safe_open(SOURCE_FILE, framework="pt") as f:
    tensor = f.get_tensor("model.embed_tokens.weight")

# Load R1 rotation matrix generated during language model compilation
rot_matrix = torch.jit.load(
    "spinWeight/Qwen2-VL-2B-Instruct_text_model/R1/global_rotation.pth"
).state_dict()["0"]

# Apply R1 rotation to text embedding
embedding = tensor.double() @ rot_matrix
save_file({"model.embed_tokens.weight": embedding.float()}, "mxq/model.safetensors")
```

**R2 (Per-layer Rotation)** applies individual rotations to each transformer layer
to compensate for weight distribution differences across layers.
R2 is absorbed into the model during MXQ compilation, so no separate post-processing is needed.

**R1 Usage in VLM**:
For VLM, the R1 generated during language model compilation is used in two places.

1. **Text embedding rotation** — Apply R1 to embedding weights, same as LLM (`vlm/get_safetensors.py`)
2. **Vision encoder alignment** — Since the vision encoder's output must match the rotated language model's input space,
   R1 is referenced at compile time via `HeadOutChRotation` (`vlm/mxq_compile_vision.py`)

No separate rotation is applied to vision embeddings.

**Usage examples**:
- `llm/generate_mxq_4bit.py` - LLM 4bit SpinQuant application
- `llm/get_rotation_emb.py` - LLM embedding R1 rotation
- `vlm/mxq_compile_language.py` - VLM language model equivalent transformation
- `vlm/mxq_compile_vision.py` - VLM vision encoder R1 rotation matrix reference
- `vlm/get_safetensors.py` - VLM text embedding R1 rotation

---

## SearchWeightScaleConfig

Learns per-layer weight scales to correct quantization accuracy.
Compilation time increases but quantization model accuracy improves.
Recommended for 4bit quantization where quantization error is large.

```python
from qbcompiler import SearchWeightScaleConfig

sws_config = SearchWeightScaleConfig(
    # Enable weight scale search.
    # Iteratively searches for optimal weight scales per layer based on calibration data
    # to minimize accuracy degradation from quantization.
    apply=True,

    transformer=SearchWeightScaleConfig.Transformer(
        # Individually specify whether to search scales for each transformer component.
        # Components set to True learn optimal scales to reduce quantization error.
        # Enabling more components improves accuracy but proportionally increases compile time.
        query=True,   # Q projection weight scale search
        key=True,     # K projection weight scale search
        value=True,   # V projection weight scale search
        out=True,     # Output projection weight scale search
        ffn=True,     # FFN weight scale search
    ),
)
```

**Usage examples**:
- `llm/generate_mxq_4bit.py`

---

## PreprocessingConfig

The compiler automatically performs image preprocessing (resize, crop, normalize) on calibration data.

With this config applied, raw images can be passed directly to `calib_data_path`
without separate preprocessing, making the quantization process more convenient.

```python
from qbcompiler import PreprocessingConfig

preprocessing_config = PreprocessingConfig(
    # Whether to enable the preprocessing pipeline
    apply=True,

    # True: Automatically detect and convert the input image's channel format (RGB/BGR, etc.).
    # Allows processing images from various sources without separate conversion.
    auto_convert_format=True,

    # Preprocessing operation pipeline. Applied in order.
    # This pipeline is fused into the model's first layer,
    # so preprocessing and inference run as a single flow on the NPU.
    pipeline=[
        # Step 1: Resize image to 256x256
        #   mode: interpolation method ("bilinear", "nearest", etc.)
        {"op": "resize", "height": 256, "width": 256, "mode": "bilinear"},

        # Step 2: Center crop to 224x224
        {"op": "centerCrop", "height": 224, "width": 224},

        # Step 3: Normalize
        {
            "op": "normalize",
            "mean": [0.485, 0.456, 0.406],  # ImageNet RGB per-channel mean
            "std": [0.229, 0.224, 0.225],    # ImageNet RGB per-channel std

            # True: Scale uint8 input ([0, 255]) to [0, 1] before normalization.
            # Allows feeding raw images directly during inference.
            "scaleToUint8": True,

            # True: Absorb normalization into the model's first layer weights.
            # Normalization is handled within the NPU without a separate preprocessing step.
            "fuseIntoFirstLayer": True,
        },
    ],
)
```

**Usage examples**:
- `image_classification/model_compile.py`

---

## Related Documents

- [Calibration Data Guide](./02_about_calibration_data.md) - Calibration data preparation for quantization
- [Multi-Component Model Guide](./03_about_multi_component.md) - Models requiring split compilation (VLM/STT)
