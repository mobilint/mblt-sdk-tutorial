import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPO_ID = "mobilint/whisper-small"
TARGET_DEVICES = ("aries-rb", "regulus-rb")


def prepare_model(target_device: str, output_dir: Path, force: bool) -> None:
    encoder_mxq = Path("mxq") / target_device / "whisper-small_encoder.mxq"
    decoder_mxq = Path("mxq") / target_device / "whisper-small_decoder.mxq"
    for path in (encoder_mxq, decoder_mxq):
        if not path.is_file():
            raise FileNotFoundError(f"MXQ file not found: {path}")

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{output_dir} already exists. Use --force to replace it.")
        shutil.rmtree(output_dir)

    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=output_dir,
        ignore_patterns=["*.mxq"],
    )
    shutil.rmtree(output_dir / ".cache", ignore_errors=True)

    encoder_name = encoder_mxq.name
    decoder_name = decoder_mxq.name
    shutil.copy2(encoder_mxq, output_dir / encoder_name)
    shutil.copy2(decoder_mxq, output_dir / decoder_name)

    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["encoder_mxq_path"] = encoder_name
    config["decoder_mxq_path"] = decoder_name
    config["encoder_core_mode"] = "single"
    config["decoder_core_mode"] = "single"
    config["encoder_target_cores"] = ["0:0"]
    config["decoder_target_cores"] = ["1:0" if target_device == "aries-rb" else "0:0"]
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Prepared model folder: {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--target-device", choices=TARGET_DEVICES, default="aries-rb")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    destination = args.output_dir or Path("prepared") / args.target_device / "whisper-small"
    prepare_model(args.target_device, destination, args.force)
