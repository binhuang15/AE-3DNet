# AE-3DNet

This repository is the official implementation of AE-3DNet, as presented in our work published in *Cancer Science*:

[**A Contrast-Enhanced Ultrasound Cine-Based Deep Learning Model for Predicting the Response of Advanced Hepatocellular Carcinoma to Hepatic Arterial Infusion Chemotherapy Combined With Systemic Therapies**](https://onlinelibrary.wiley.com/doi/full/10.1111/cas.70089)

## Table of Contents
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Data Preparation](#step-1-data-preparation)
  - [Step 2: Model Training](#step-2-model-training)
  - [Step 3: Model Evaluation](#step-3-model-evaluation)

## Repository Structure

* [`kfold_train.py`](./kfold_train.py): Script for model training (using k-fold cross-validation setup).
* [`kfold_test.py`](./kfold_test.py): Script for model testing and evaluation.
* [`model.py`](./model.py): Contains the implementation of our AE-3DNet model.

## Getting Started

Follow these steps to set up the environment and reproduce the results.

### Prerequisites

Our code was implemented in the following environment:

* Python == 3.7.0
* torch == 1.7.0
* cuda == 11.0
* GPU == NVIDIA GTX 1080Ti (11GB memory) or equivalent

### Installation

1.  We recommend creating a virtual environment (e.g., using `conda` or `venv`):
    ```bash
    conda create -n ae3dnet python=3.7
    conda activate ae3dnet
    ```
2.  Install the required libraries. You can install them directly or create a `requirements.txt` file.
    
    A `requirements.txt` file based on the dependencies would look like this:
    ```txt
    torch==1.7.0
    numpy
    # Add other libraries like scikit-learn, pillow, etc. if used
    ```
    Then install using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This code can be easily performed on CEUS data with classification notation.

### Step 1: Data Preparation

1.  Prepare the data according to our paper, ensuring that the data for each patient comprise the required number of CEUS image crops.
2.  Split the patients into five folds to yield a training set, a validation set, and a test set.
3.  Save the data path and the corresponding label in a `.npy` file for each set (`train.npy`, `val.npy`, `test.npy`). The format should be a list of lists, where each inner list is `[path_to_data, label]`.

    **Example `.npy` file content:**
    ```python
    # train.npy
    [["train_path_1", "0"], 
     ["train_path_2", "1"], 
     ..., 
     ["train_path_N", "0"]]
    
    # val.npy
    [["val_path_1", "1"], 
     ["val_path_2", "0"], 
     ..., 
     ["val_path_N", "0"]]
    
    # test.npy
    [["test_path_1", "1"], 
     ["test_path_2", "0"], 
     ..., 
     ["test_path_N", "1"]]
    ```

4.  Organize these `.npy` files into a directory structure for the 5-fold cross-validation. The code expects a root folder (e.g., `ALL_Data`) containing subfolders named `0` through `4`:

    ```
    ALL_Data/
    ├── 0/
    │   ├── train.npy
    │   ├── val.npy
    │   └── test.npy
    ├── 1/
    │   ├── train.npy
    │   ├── val.npy
    │   └── test.npy
    ├── 2/
    │   ├── ...
    ├── 3/
    │   ├── ...
    └── 4/
        ├── train.npy
        ├── val.npy
        └── test.npy
    ```

### Step 2: Model Training

Run the script [`kfold_train.py`](./kfold_train.py) to train AE-3DNet. This script will use the `train.npy` (for training) and `val.npy` (for model selection) files from each fold.

```bash
python kfold_train.py --data_path ./ALL_Data
```
During training, checkpoints with the best performance on the validation set will be saved in the `model_checkpoint` folder.

### Step 3: Model Evaluation

Run the script [`kfold_test.py`](./kfold_test.py) to evaluate the model's performance on the `test.npy` files.

1.  Before running, select the best checkpoint from each fold (saved in `model_checkpoint`) and move them into a new folder (e.g., `model_for_test`).

2.  Run the test script:

```bash
python kfold_test.py --data_path ./ALL_Data --model_save_name ./model_for_test
```

This script will output the ROC curve and the corresponding AUC of AE-3DNet on the test set.
