# Large Language Model

This tutorial provides detailed instructions for compiling large language models using the Mobilint QB compiler.

In this tutorial, we will use the [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model, a large language model developed by Meta.

## Model Preparation

First, we need to prepare the model.

Before using the model, sign up for an account on [HuggingFace](https://huggingface.co/) and sign the agreement to use the model on the [model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).

In addition, visit the [access token management page](https://huggingface.co/settings/tokens) and get an access token. A read-only token is sufficient for downloading the model, so create one if it doesn't exist.

Go back to the terminal and set the access token via the HuggingFace CLI:

```bash
pip install huggingface_hub[cli]
hf auth login {your_access_token}
```

Then, download the model using the following commands:

```bash
apt-get install git-lfs # Install git-lfs if not installed
git lfs install # Initialize git-lfs
git clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
```

## Calibration Dataset Preparation

Before downloading the dataset and creating a calibration dataset, we need to extract the embedding weights from the model. Since the calibration dataset must be in a format that the quantized model can handle, and the embedding weights will not be included in the quantized model, we need to extract them from the original model first.

```bash
python3 get_embedding_weight.py # Extract the embedding weights from the model
```

After execution, the embedding weights are saved as `Llama-3.2-1B-Instruct_embedding_weight.pt` in the current directory.

LLMs like Llama-3.2-1B-Instruct are trained on a massive amount of text data, so it is difficult to specify a specific dataset for calibration. In this tutorial, we will use the [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) dataset for calibration.

To enable dataset downloading, we need to install the `datasets` library:

```bash
pip install datasets # Install datasets library if not installed
```

Then, create a calibration dataset using the following command:

```bash
python3 prepare_calib.py --lang {language} --min_seqlen {sequence_length_minimum} --max_seqlen {sequence_length_maximum} --max_calib {maximum_number_of_calibration}
```

The script `prepare_calib.py` downloads the Wikipedia dataset, tokenizes the text, and embeds the text into a tensor. The tensor is saved as `npy` files in the `calib_dir` directory.

The example command is as follows:

```bash
python3 prepare_calib.py --lang en --min_seqlen 512 --max_seqlen 2048 --max_calib 128
```

The calibration dataset is saved as `Llama-3.2-1B-Instruct-Wikipedia-en` in the current directory.

> Note: After execution, the downloaded dataset may continue to occupy disk space. You can remove it by running the following command:
> ```bash
> hf cache delete
> ```

## Model Compilation

After the calibration dataset and the model are prepared, we can compile the model.

```bash
python3 model_compile.py --model_path {path_to_model} --calib_data_path {path_to_calibration_dataset} --save_path {path_to_save_model}
```

The example command is as follows:

```bash
python3 model_compile.py --model_path ./Llama-3.2-1B-Instruct --calib_data_path ./Llama-3.2-1B-Instruct-Wikipedia-en --save_path ./Llama-3.2-1B-Instruct.mxq
```

The compiled model is saved as `Llama-3.2-1B-Instruct.mxq` in the current directory.
