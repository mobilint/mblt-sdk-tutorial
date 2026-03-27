import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file
from transformers import (
    GenerationConfig,
    WhisperConfig,
    WhisperForConditionalGeneration,
)


def prepare_model_folder(
    encoder_mxq_path: str,
    decoder_mxq_path: str,
    output_folder: str,
    base_model: str = "openai/whisper-small",
    model_id: str = "mobilint/whisper-small",
):
    """
    Prepare a model folder with proper configuration for MXQ inference.

    Args:
        encoder_mxq_path: Path to the compiled encoder MXQ file
        decoder_mxq_path: Path to the compiled decoder MXQ file
        output_folder: Output folder to create with all necessary files
        base_model: HuggingFace model ID for base configuration and embedding extraction
        model_id: HuggingFace model ID for mblt-model-zoo (stored in config._name_or_path)
    """
    print(f"Preparing model folder: {output_folder}")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Copy MXQ files to output folder
    encoder_filename = os.path.basename(encoder_mxq_path)
    decoder_filename = os.path.basename(decoder_mxq_path)

    print(f"Copying encoder MXQ: {encoder_mxq_path}")
    shutil.copy(encoder_mxq_path, os.path.join(output_folder, encoder_filename))

    print(f"Copying decoder MXQ: {decoder_mxq_path}")
    shutil.copy(decoder_mxq_path, os.path.join(output_folder, decoder_filename))

    # Load base configuration
    print(f"Loading base configuration from {base_model}...")
    base_config = WhisperConfig.from_pretrained(base_model)

    # Create Mobilint Whisper configuration
    config_dict = base_config.to_dict()
    config_dict["_name_or_path"] = model_id
    config_dict["model_type"] = "mobilint-whisper"
    config_dict["encoder_mxq_path"] = encoder_filename
    config_dict["decoder_mxq_path"] = decoder_filename
    config_dict["dev_no"] = 0
    # NPU core allocation for mblt-model-zoo inference.
    # Format: "cluster:core" (e.g., "0:0" = Cluster0, Core0)
    # Required when MXQ is compiled with inference_scheme="all".
    #
    # Available core modes (same fields apply to decoder with decoder_ prefix):
    #   single: encoder_target_cores=["0:0"]    — Cluster 0, Core 0 (default)
    #           encoder_target_cores=["0:1"]    — Cluster 0, Core 1
    #           encoder_target_cores=["0:3"]    — Cluster 0, Core 3
    #           encoder_target_cores=["1:0"]    — Cluster 1, Core 0
    #   multi:   encoder_core_mode="multi",   encoder_target_clusters=[0]
    #   global4: encoder_core_mode="global4", encoder_target_clusters=[0]
    #   global8: encoder_core_mode="global8", encoder_target_clusters=[0, 1]
    config_dict["encoder_target_cores"] = ["0:0"]
    config_dict["decoder_target_cores"] = ["0:0"]
    config_dict["architectures"] = ["MobilintWhisperForConditionalGeneration"]

    # Save configuration
    config_path = os.path.join(output_folder, "config.json")
    print(f"Saving configuration to {config_path}")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save generation config
    gen_config = GenerationConfig.from_pretrained(base_model)
    gen_config.save_pretrained(output_folder)

    # Extract and save embedding weights from the base model
    print(f"Extracting embedding weights from {base_model}...")
    base_model_obj = WhisperForConditionalGeneration.from_pretrained(
        base_model, dtype=torch.float32
    )

    # Get decoder embedding weights and positional embeddings
    embed_tokens_weight = base_model_obj.model.decoder.embed_tokens.weight.data
    embed_positions_weight = base_model_obj.model.decoder.embed_positions.weight.data

    # Save only the embedding weights to safetensors
    embedding_tensors = {
        "model.decoder.embed_tokens.weight": embed_tokens_weight,
        "model.decoder.embed_positions.weight": embed_positions_weight,
    }

    safetensors_path = os.path.join(output_folder, "model.safetensors")
    print(f"Saving embedding weights to {safetensors_path}")
    save_file(embedding_tensors, safetensors_path)

    # Clean up to free memory
    del base_model_obj

    print(f"\nModel folder prepared successfully: {output_folder}")
    print("\nFiles in output folder:")
    for f in sorted(os.listdir(output_folder)):
        filepath = os.path.join(output_folder, f)
        size = os.path.getsize(filepath)
        if size > 1024 * 1024:
            print(f"  {f} ({size / 1024 / 1024:.2f} MB)")
        elif size > 1024:
            print(f"  {f} ({size / 1024:.2f} KB)")
        else:
            print(f"  {f} ({size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Model Folder for Whisper MXQ Inference")
    parser.add_argument(
        "--encoder-mxq",
        type=str,
        default="../../compilation/stt/mxq/whisper-small_encoder.mxq",
        help="Path to the compiled encoder MXQ file",
    )
    parser.add_argument(
        "--decoder-mxq",
        type=str,
        default="../../compilation/stt/mxq/whisper-small_decoder.mxq",
        help="Path to the compiled decoder MXQ file",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./whisper-small-mxq",
        help="Output folder to create with all necessary files",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="openai/whisper-small",
        help="HuggingFace model ID for base configuration and embedding extraction",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="mobilint/whisper-small",
        help="HuggingFace model ID for mblt-model-zoo (stored in config._name_or_path)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.encoder_mxq):
        raise FileNotFoundError(f"Encoder MXQ file not found: {args.encoder_mxq}")

    if not os.path.exists(args.decoder_mxq):
        raise FileNotFoundError(f"Decoder MXQ file not found: {args.decoder_mxq}")

    prepare_model_folder(
        encoder_mxq_path=args.encoder_mxq,
        decoder_mxq_path=args.decoder_mxq,
        output_folder=args.output_folder,
        base_model=args.base_model,
        model_id=args.model_id,
    )

    print("\nYou can now run inference with:")
    print(
        f"  python inference_mblt_model_zoo.py --audio /path/to/audio.wav --model-folder {args.output_folder}"
    )
