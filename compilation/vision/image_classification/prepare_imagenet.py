import os

from datasets import load_dataset
from tqdm import tqdm

if __name__ == "__main__":
    data_files = {"validation": "data/validation*.parquet"}
    dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        data_files=data_files,
        split="validation",
        verification_mode="no_checks",
    )  # download the dataset from HuggingFace

    os.makedirs("imagenet-1k-selected", exist_ok=True)

    labels = list(range(1000))
    total_labels = len(labels)
    pbar = tqdm(total=total_labels, desc="Finding labels", unit="label")

    for sample in dataset:
        if sample["label"] in labels:
            image = sample["image"].convert("RGB")
            image.save(f"imagenet-1k-selected/{sample['label']}.JPEG")
            labels.remove(sample["label"])
            pbar.update(1)

        if len(labels) == 0:
            break

    pbar.close()
