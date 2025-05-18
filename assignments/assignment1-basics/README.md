# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# download the data from the HF mirror
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

# download the data from the HF mirror
wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### Training my Transformer Language Model

1. **Prepare your data and tokenizer:**
   - Ensure you have downloaded and preprocessed your data (see the "Download data" section above).
   - Train a BPE tokenizer if you haven't already:
     ```sh
     uv run cs336_basics/train_bpe_tinystories.py
     ```
     or for OpenWebText:
     ```sh
     uv run cs336_basics/train_bpe_owt.py
     ```

2. **Tokenize your dataset (if required):**
   - Use your tokenizer to convert raw text into token IDs. (Refer to the assignment handout or provided scripts for details.)

3. **Train the Transformer LM:**
   - Run the training script. For TinyStories, for example:
     ```sh
     uv run cs336_basics/train_tiny_stories.py --data_dir ./data --output_dir ./output
     ```
   - You can adjust model hyperparameters (like `--context_length`, `--d_model`, `--num_layers`, etc.) using command-line arguments. See the script for all options.

4. **Monitor training:**
   - Training logs, checkpoints, and (optionally) wandb logs will be saved in the output directory you specify.

5. **Evaluate and generate text:**
   - After training, you can use the provided `generate.py` script to generate text from your trained model:
     ```sh
     uv run cs336_basics/generate.py --model_path ./output/checkpoints/your_model.pt --prompt "Once upon a time"
     ```

**Quality Tips:**
- Make sure your tokenizer and model hyperparameters match.
- Use a validation set to monitor for overfitting.
- For best results, train for enough iterations and tune your learning rate and batch size according to your hardware.

