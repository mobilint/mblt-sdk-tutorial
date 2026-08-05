import argparse
import glob
import json
import os
import shutil

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "mobilint/whisper-small"


def download_repo(repo_id: str, output_folder: str, force: bool) -> None:
    """Download the self-contained HF repo, skipping only its `.mxq` (swapped in next step).
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
        ignore_patterns=["*.mxq"],
    )


def replace_artifacts(compilation_dir: str, output_folder: str) -> tuple[str, str]:
    """Swap in the compiled encoder/decoder MXQ (removing stale `.mxq` first).

    Returns (encoder_mxq, decoder_mxq) filenames.
    """
    for path in glob.glob(os.path.join(output_folder, "*.mxq")):
        print(f"Removing old artifact: {os.path.basename(path)}")
        os.remove(path)

    mxq_files = sorted(f for f in os.listdir(compilation_dir) if f.endswith(".mxq"))
    if len(mxq_files) < 2:
        raise FileNotFoundError(f"Expected 2 .mxq in {compilation_dir}, found {mxq_files}")

    encoder_mxq = next((f for f in mxq_files if "encoder" in f.lower()), None)
    decoder_mxq = next((f for f in mxq_files if "decoder" in f.lower()), None)
    if encoder_mxq is None or decoder_mxq is None:
        raise ValueError(f"Could not classify encoder/decoder MXQ from {mxq_files}")

    for filename in (encoder_mxq, decoder_mxq):
        print(f"Copying: {filename}")
        shutil.copy(os.path.join(compilation_dir, filename), os.path.join(output_folder, filename))

    return encoder_mxq, decoder_mxq


def patch_config(output_folder: str, encoder_mxq: str, decoder_mxq: str) -> None:
    """Point config.json's mxq paths at the copied files; keep the repo's core allocation."""
    config_path = os.path.join(output_folder, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    config["encoder_mxq_path"] = encoder_mxq
    config["decoder_mxq_path"] = decoder_mxq

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Patched config.json: encoder_mxq_path={encoder_mxq}, decoder_mxq_path={decoder_mxq}")


def prepare_model_folder(repo_id: str, compilation_dir: str, output_folder: str, force: bool) -> None:
    """Build a self-contained model folder: download repo, swap in compiled MXQ, patch config."""
    download_repo(repo_id, output_folder, force)
    encoder_mxq, decoder_mxq = replace_artifacts(compilation_dir, output_folder)
    patch_config(output_folder, encoder_mxq, decoder_mxq)

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
    parser = argparse.ArgumentParser(description="Prepare a self-contained Whisper MXQ model folder")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID,
                        help="HuggingFace repo id to download (self-contained config/proxy/tokenizer/embeddings).")
    parser.add_argument("--compilation-dir", type=str, default="../../../compilation/stt/mxq",
                        help="Compilation output dir holding the 2 .mxq (encoder and decoder).")
    parser.add_argument("--output-folder", type=str, default="./whisper-small-mxq",
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
    print(f"  python inference_mblt_model_zoo.py --audio /path/to/audio.wav --model-folder {args.output_folder}")
