import argparse
import glob
import json
import os
import shutil
import subprocess

DEFAULT_REPO_URL = "https://huggingface.co/mobilint/Qwen3-VL-4B-Instruct"


def clone_repo(repo_url: str, output_folder: str, force: bool) -> None:
    """git clone the HF repo into output_folder (self-contained: config, proxy, tokenizer).

    Requires git-lfs so the large tracked files (tokenizer, etc.) are real files,
    not LFS pointers.
    """
    if os.path.exists(output_folder):
        if not force:
            raise FileExistsError(
                f"{output_folder} already exists. Use --force to remove and re-clone."
            )
        print(f"Removing existing folder: {output_folder}")
        shutil.rmtree(output_folder)

    print(f"Cloning {repo_url} -> {output_folder}")
    subprocess.run(["git", "clone", repo_url, output_folder], check=True)


def replace_artifacts(compilation_dir: str, output_folder: str) -> tuple[str, str]:
    """Delete the repo's old .mxq/.safetensors and copy the freshly compiled ones.

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

    Core allocation (core_mode / target_cores) from the cloned repo is left as-is.
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


def prepare_model_folder(repo_url: str, compilation_dir: str, output_folder: str, force: bool) -> None:
    """Build a self-contained model folder: clone repo, swap in compiled MXQ, patch config."""
    clone_repo(repo_url, output_folder, force)
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
    parser.add_argument("--repo-url", type=str, default=DEFAULT_REPO_URL,
                        help="HuggingFace repo to clone (self-contained config/proxy/tokenizer).")
    parser.add_argument("--compilation-dir", type=str, default="../../../compilation/vlm/mxq",
                        help="Compilation output dir holding the 2 .mxq and 1 .safetensors.")
    parser.add_argument("--output-folder", type=str, default="./Qwen3-VL-4B-Instruct",
                        help="Destination folder (cloned repo + swapped-in compiled artifacts).")
    parser.add_argument("--force", action="store_true",
                        help="Remove output-folder first if it already exists.")
    args = parser.parse_args()

    if not os.path.exists(args.compilation_dir):
        raise FileNotFoundError(
            f"Compilation directory not found: {args.compilation_dir}\n"
            f"Please run the compilation tutorial first."
        )

    prepare_model_folder(
        repo_url=args.repo_url,
        compilation_dir=args.compilation_dir,
        output_folder=args.output_folder,
        force=args.force,
    )

    print("\nYou can now run inference with:")
    print(f"  python inference_mblt_model_zoo.py --model-folder {args.output_folder}")
