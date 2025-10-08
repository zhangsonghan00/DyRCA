# DyRCA: A Dynamic-Aware Framework for Root Cause Analysis in Microservice Systems

This repository provides supplemental material (including source code and data) for the paper **"DyRCA: A Dynamic-Aware Framework for Root Cause Analysis in Microservice Systems"**. Follow the steps below to replicate the experimental results presented in the paper.


## 1. Environment Setup

We recommend using **uv** (a fast Python package and environment manager) for environment configuration. Alternatively, you can use `pip` if you prefer.


### Using uv (Recommended)

First, install uv and configure the project environment:
```bash
# Step 1: Install uv (cross-platform, Linux/macOS compatible)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Step 2: Install all project dependencies (reads pyproject.toml)
uv sync

# Step 3: Activate the virtual environment
source .venv/bin/activate  # Linux/macOS

```

### Using pip

If you choose pip, use the following commands to install dependencies.

```bash
# Step 1: Create and activate a virtual environment with venv
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Step 2: Install project dependencies (including the package itself)
pip install .
```

## 2. View Command Help

To check all available commands and parameters (for both uv and pip), run:

```bash

# Using uv
uv run client.py --help

# Using pip
python client.py --help

```

## 3. Data Preprocessing

We provide preprocessed data directly usable for training/testing. If you need to reproduce the preprocessing from raw datasets:
Download the original public datasets (references provided in the paper).
Run the preprocessing command below, specifying the dataset type.

```bash
# Using uv
uv run client.py preprocess --dataset-type <dataset_name>
# Replace <dataset_name> with the actual dataset identifier (e.g., RCAEval_re2_ob)

# Using pip
python client.py preprocess --dataset-type <dataset_name>

```

## 4. Model Training

Use the following commands to train the DyRCA model. Adjust hyperparameters (e.g., epochs, hidden dimension) as needed.

```bash
# Using uv
uv run client.py train --epochs <epochs> --hidden-dim <hidden_dim>
# Example: uv run client.py train --epochs 30 --hidden-dim 32

# Using pip
python client.py train --epochs <epochs> --hidden-dim <hidden_dim>
# Example: python client.py train --epochs 30 --hidden-dim 32

```

## 5. Model Testing

After training, evaluate the model using the preprocessed test data. Specify the path to the trained model checkpoint.

```bash
# Using uv
uv run client.py test --model_path <model-path>
# Replace <model-path> with the actual path to the trained model (e.g., "./dyrca_best.pth")

# Using pip
python client.py test --model_path <model-path>

```
