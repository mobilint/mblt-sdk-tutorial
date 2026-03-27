import argparse
import json
import os
import shutil


def prepare_model_folder(
    compilation_dir: str,
    output_folder: str,
    model_id: str,
):
    """
    Prepare a model folder for VLM MXQ inference.

    Args:
        compilation_dir: Path to the compilation output directory containing MXQ files
        output_folder: Destination folder for the prepared model
        model_id: HuggingFace model ID stored in config for mblt-model-zoo model registration
    """
    os.makedirs(output_folder, exist_ok=True)

    # Copy all files from compilation output
    required_files = [
        "config.json",
        "model.safetensors",
    ]

    # Find MXQ files
    mxq_files = [f for f in os.listdir(compilation_dir) if f.endswith(".mxq")]
    if len(mxq_files) < 2:
        raise FileNotFoundError(
            f"Expected at least 2 MXQ files in {compilation_dir}, found {len(mxq_files)}"
        )

    all_files = required_files + mxq_files
    for filename in all_files:
        src = os.path.join(compilation_dir, filename)
        dst = os.path.join(output_folder, filename)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Required file not found: {src}")
        print(f"Copying: {filename}")
        shutil.copy(src, dst)

    # Add NPU core allocation to config.json
    # Required when MXQ is compiled with inference_scheme="all".
    #
    # Available core modes (vision encoder uses same fields under vision_config):
    #   single: target_cores=["0:0"]    — Cluster 0, Core 0 (default)
    #           target_cores=["0:1"]    — Cluster 0, Core 1
    #           target_cores=["0:3"]    — Cluster 0, Core 3
    #           target_cores=["1:0"]    — Cluster 1, Core 0
    #   multi:   core_mode="multi",   target_clusters=[0]
    #   global4: core_mode="global4", target_clusters=[0]
    #   global8: core_mode="global8", target_clusters=[0, 1]
    config_path = os.path.join(output_folder, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    config["_name_or_path"] = model_id
    config["target_cores"] = ["0:0"]
    if "vision_config" in config:
        config["vision_config"]["target_cores"] = ["0:0"]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print("Updated config.json with NPU core allocation")

    # Print summary
    print(f"\nModel folder prepared: {output_folder}")
    print("Contents:")
    for f in sorted(os.listdir(output_folder)):
        size = os.path.getsize(os.path.join(output_folder, f))
        if size > 1024 * 1024:
            print(f"  {f} ({size / 1024 / 1024:.2f} MB)")
        elif size > 1024:
            print(f"  {f} ({size / 1024:.2f} KB)")
        else:
            print(f"  {f} ({size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Model Folder for VLM MXQ Inference")
    parser.add_argument(
        "--compilation-dir",
        type=str,
        default="../../compilation/vlm/mxq",
        help="Path to the compilation output directory",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="./qwen2-vl-mxq",
        help="Destination folder for the prepared model",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="mobilint/Qwen2-VL-2B-Instruct",
        help="HuggingFace model ID stored in config for mblt-model-zoo model registration",
    )

    args = parser.parse_args()

    if not os.path.exists(args.compilation_dir):
        raise FileNotFoundError(
            f"Compilation directory not found: {args.compilation_dir}\n"
            "Please run the compilation tutorial first."
        )

    prepare_model_folder(
        compilation_dir=args.compilation_dir,
        output_folder=args.output_folder,
        model_id=args.model_id,
    )

    print("\nYou can now run inference with:")
    print(
        f"  python inference_mblt_model_zoo.py --model-folder {args.output_folder}"
    )
