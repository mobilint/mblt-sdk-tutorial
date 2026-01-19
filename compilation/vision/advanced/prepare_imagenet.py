import os
import random

random.seed(42)

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

    os.makedirs("./imagenet-1k-1000cls-1000", exist_ok=True)

    labels = list(range(1000))
    total_labels = len(labels)
    pbar = tqdm(total=total_labels, desc="Finding labels", unit="label")

    for sample in dataset:
        if sample["label"] in labels:
            image = sample["image"]
            image.save(f"./imagenet-1k-1000cls-1000/{sample['label']}.jpg")
            labels.remove(sample["label"])
            pbar.update(1)

        if len(labels) == 0:
            break

    pbar.close()

    os.makedirs("./imagenet-1k-100cls-100", exist_ok=True)

    labels = random.sample(range(1000), 100)
    total_labels = len(labels)
    pbar = tqdm(total=total_labels, desc="Finding labels", unit="label")

    for sample in dataset:
        if sample["label"] in labels:
            image = sample["image"]
            image.save(f"./imagenet-1k-100cls-100/{sample['label']}.jpg")
            labels.remove(sample["label"])
            pbar.update(1)

        if len(labels) == 0:
            break

    pbar.close()

    os.makedirs("./imagenet-1k-10cls-10", exist_ok=True)

    labels = random.sample(range(1000), 10)
    total_labels = len(labels)
    pbar = tqdm(total=total_labels, desc="Finding labels", unit="label")

    for sample in dataset:
        if sample["label"] in labels:
            image = sample["image"]
            image.save(f"./imagenet-1k-10cls-10/{sample['label']}.jpg")
            labels.remove(sample["label"])
            pbar.update(1)

        if len(labels) == 0:
            break

    pbar.close()

    os.makedirs("./imagenet-1k-5cls-100", exist_ok=True)

    labels = random.sample(range(1000), 5)
    label_count = {label: 0 for label in labels}
    total_labels = len(labels)
    pbar = tqdm(total=total_labels, desc="Finding labels", unit="label")

    for sample in dataset:
        if sample["label"] in labels:
            image = sample["image"]
            image.save(
                f"./imagenet-1k-5cls-100/{sample['label']}_{label_count[sample['label']]}.jpg"
            )
            label_count[sample["label"]] += 1
            if label_count[sample["label"]] == 20:
                labels.remove(sample["label"])
                pbar.update(1)

        if len(labels) == 0:
            break

    pbar.close()

    os.makedirs("./imagenet-1k-20cls-100", exist_ok=True)

    labels = random.sample(range(1000), 20)
    label_count = {label: 0 for label in labels}
    total_labels = len(labels)
    pbar = tqdm(total=total_labels, desc="Finding labels", unit="label")

    for sample in dataset:
        if sample["label"] in labels:
            image = sample["image"]
            image.save(
                f"./imagenet-1k-20cls-100/{sample['label']}_{label_count[sample['label']]}.jpg"
            )
            label_count[sample["label"]] += 1
            if label_count[sample["label"]] == 5:
                labels.remove(sample["label"])
                pbar.update(1)

        if len(labels) == 0:
            break

    pbar.close()
