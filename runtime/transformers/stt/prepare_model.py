#!/usr/bin/env python3
"""
Prepare Model Folder for Whisper MXQ Inference

This script prepares a model folder with the necessary configuration files
for running inference with compiled Whisper MXQ models.
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def prepare_model_folder(
    encoder_mxq_path: str,
    decoder_mxq_path: str,
    output_folder: str,
    base_model: str = "openai/whisper-small",
):
    """
    Prepare a model folder with proper configuration for MXQ inference.

    Args:
        encoder_mxq_path: Path to the compiled encoder MXQ file
        decoder_mxq_path: Path to the compiled decoder MXQ file
        output_folder: Output folder to create with all necessary files
        base_model: HuggingFace model ID to get base configuration from
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
    config_dict["_name_or_path"] = output_folder
    config_dict["model_type"] = "mobilint-whisper"
    config_dict["encoder_mxq_path"] = encoder_filename
    config_dict["decoder_mxq_path"] = decoder_filename
    config_dict["dev_no"] = 0
    config_dict["architectures"] = ["MobilintWhisperForConditionalGeneration"]

    # Save configuration
    config_path = os.path.join(output_folder, "config.json")
    print(f"Saving configuration to {config_path}")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Download and save processor files (tokenizer + feature extractor)
    print(f"Downloading processor from {base_model}...")
    processor = WhisperProcessor.from_pretrained(base_model)
    processor.save_pretrained(output_folder)

    # Save generation config
    from transformers import GenerationConfig

    gen_config = GenerationConfig.from_pretrained(base_model)
    gen_config.save_pretrained(output_folder)

    # Extract and save embedding weights from the base model
    print(f"Extracting embedding weights from {base_model}...")
    base_model_obj = WhisperForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.float32
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
    print(f"\nFiles in output folder:")
    for f in sorted(os.listdir(output_folder)):
        filepath = os.path.join(output_folder, f)
        size = os.path.getsize(filepath)
        if size > 1024 * 1024:
            print(f"  {f} ({size / 1024 / 1024:.2f} MB)")
        elif size > 1024:
            print(f"  {f} ({size / 1024:.2f} KB)")
        else:
            print(f"  {f} ({size} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Model Folder for Whisper MXQ Inference"
    )
    parser.add_argument(
        "--encoder-mxq",
        type=str,
        default="../../../compilation/transformers/stt/compilation/compiled/whisper-small_encoder.mxq",
        help="Path to the compiled encoder MXQ file",
    )
    parser.add_argument(
        "--decoder-mxq",
        type=str,
        default="../../../compilation/transformers/stt/compilation/compiled/whisper-small_decoder.mxq",
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
        help="HuggingFace model ID to get base configuration from",
    )

    args = parser.parse_args()

    # Verify MXQ files exist
    if not os.path.exists(args.encoder_mxq):
        print(f"Error: Encoder MXQ file not found: {args.encoder_mxq}")
        print("Please run the compilation tutorial first.")
        return 1

    if not os.path.exists(args.decoder_mxq):
        print(f"Error: Decoder MXQ file not found: {args.decoder_mxq}")
        print("Please run the compilation tutorial first.")
        return 1

    prepare_model_folder(
        encoder_mxq_path=args.encoder_mxq,
        decoder_mxq_path=args.decoder_mxq,
        output_folder=args.output_folder,
        base_model=args.base_model,
    )

    print("\nYou can now run inference with:")
    print(
        f"  python inference_mxq.py --audio /path/to/audio.wav --model_folder {args.output_folder}"
    )

    return 0


if __name__ == "__main__":
    exit(main())
