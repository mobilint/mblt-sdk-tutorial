import argparse
import glob
import json
import os
import shutil

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "mobilint/Qwen3-VL-2B-Instruct"


def download_repo(repo_id: str, output_folder: str, force: bool) -> None:
    """Download the HF repo into output_folder (self-contained: config, proxy, tokenizer).

    The repo's own `.mxq` / `.safetensors` are skipped — they are replaced with the
    freshly compiled artifacts in the next step.
    """
    if os.path.exists(output_folder):
        if not force:
            raise FileExistsError(
                f"{output_folder} already exists. Use --force to remove and re-download."
            )
        print(f"Removing existing folder: {output_folder}")
        shutil.rmtree(output_folder)

    print(f"Downloading {repo_id} -> {output_folder}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=output_folder,
        ignore_patterns=["*.mxq", "*.safetensors"],
    )


def replace_artifacts(compilation_dir: str, output_folder: str) -> tuple[str, str]:
    """Copy the freshly compiled MXQ/safetensors into the folder.

    Any leftover `.mxq` / `.safetensors` in the folder are removed first.
    Returns (vision_mxq_filename, text_mxq_filename).
    """
    for pattern in ("*.mxq", "*.safetensors"):
        for path in glob.glob(os.path.join(output_folder, pattern)):
            print(f"Removing old artifact: {os.path.basename(path)}")
            os.remove(path)

    mxq_files = sorted(f for f in os.listdir(compilation_dir) if f.endswith(".mxq"))
    safetensors = sorted(f for f in os.listdir(compilation_dir) if f.endswith(".safetensors"))
    if len(mxq_files) < 2:
        raise FileNotFoundError(f"Expected 2 .mxq in {compilation_dir}, found {mxq_files}")
    if not safetensors:
        raise FileNotFoundError(f"Expected 1 .safetensors in {compilation_dir}, found none")

    for filename in mxq_files + safetensors[:1]:
        print(f"Copying: {filename}")
        shutil.copy(os.path.join(compilation_dir, filename), os.path.join(output_folder, filename))

    vision_mxq = next((f for f in mxq_files if "vision" in f.lower()), None)
    text_mxq = next((f for f in mxq_files if "text" in f.lower() or "language" in f.lower()), None)
    if vision_mxq is None or text_mxq is None:
        raise ValueError(f"Could not classify vision/text MXQ from {mxq_files}")
    return vision_mxq, text_mxq


def patch_config(output_folder: str, vision_mxq: str, text_mxq: str) -> None:
    """Point config.json's mxq_path fields at the copied MXQ files.

    Core allocation (core_mode / target_cores) from the downloaded repo is left as-is.
    """
    config_path = os.path.join(output_folder, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Some compiled configs keep the language-model path at top level; normalize.
    config.pop("mxq_path", None)
    config["text_config"]["mxq_path"] = text_mxq
    config["vision_config"]["mxq_path"] = vision_mxq

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Patched config.json: text_config.mxq_path={text_mxq}, vision_config.mxq_path={vision_mxq}")


def prepare_model_folder(repo_id: str, compilation_dir: str, output_folder: str, force: bool) -> None:
    """Build a self-contained model folder: download repo, swap in compiled MXQ, patch config."""
    download_repo(repo_id, output_folder, force)
    vision_mxq, text_mxq = replace_artifacts(compilation_dir, output_folder)
    patch_config(output_folder, vision_mxq, text_mxq)

    print(f"\nModel folder prepared: {output_folder}")
    print("Contents:")
    for name in sorted(os.listdir(output_folder)):
        path = os.path.join(output_folder, name)
        if os.path.isdir(path):
            continue
        size = os.path.getsize(path)
        if size > 1024 * 1024:
            print(f"  {name} ({size / 1024 / 1024:.2f} MB)")
        elif size > 1024:
            print(f"  {name} ({size / 1024:.2f} KB)")
        else:
            print(f"  {name} ({size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a self-contained VLM MXQ model folder")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID,
                        help="HuggingFace repo id to download (self-contained config/proxy/tokenizer).")
    parser.add_argument("--compilation-dir", type=str, default="../../../compilation/vlm/mxq",
                        help="Compilation output dir holding the 2 .mxq and 1 .safetensors.")
    parser.add_argument("--output-folder", type=str, default="./Qwen3-VL-2B-Instruct",
                        help="Destination folder (downloaded repo + swapped-in compiled artifacts).")
    parser.add_argument("--force", action="store_true",
                        help="Remove output-folder first if it already exists.")
    args = parser.parse_args()

    if not os.path.exists(args.compilation_dir):
        raise FileNotFoundError(
            f"Compilation directory not found: {args.compilation_dir}\n"
            f"Please run the compilation tutorial first."
        )

    prepare_model_folder(
        repo_id=args.repo_id,
        compilation_dir=args.compilation_dir,
        output_folder=args.output_folder,
        force=args.force,
    )

    print("\nYou can now run inference with:")
    print(f"  python inference_mblt_model_zoo.py --model-folder {args.output_folder}")
