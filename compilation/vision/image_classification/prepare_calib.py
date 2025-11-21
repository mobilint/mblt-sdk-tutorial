from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation")

    print(dataset.keys())
