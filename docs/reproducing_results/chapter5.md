This section presents how the results presented in Chapter 5 can be reproduced. 
The corresponding scripts are all located in the `unsupervised_anomaly_detection` folder.

First, download the data.
This chapter uses two different datasets: the PATH dataset and a proprietary real-world dataset.

## PATH Dataset

### 1. Parsing
The PATH dataset is hosted on Zenodo and can be downloaded [here](https://zenodo.org/records/13255120).
Once downloaded, unzip either or both `0_simulation` and `1_postsim` folders to `data/path`.

If you only downloaded `0_simulation`, you need to parse it first using the `path/0_postsim.py` script.
The output will be identical to the contents of the `1_postsim` folder.

### 2. Preprocessing
Then, preprocess the contents of the `1_postsim` folder using `path/1_data.py`, which includes downsampling, standardisation, and windowing.
By default, the data is preprocessed for **unsupervised** anomaly detection, as set by the variable `AD_MODE = "us"`.
For **semi-supervised** anomaly detection, set `AD_MODE = "ss"`.

### 3. Training
Once preprocessed, the model training for different experiments can be run with the `path/2_training.py` script.
Like in `path/1_data.py`, the `AD_MODE` variable denotes whether the **unsupervised** or the **semi-supervised** version of the PATH dataset is used.
The difference between them is that the training and validation subsets of the **unsupervised** version contain unlabelled anomalous sequences, whereas the **semi-supervised** version contains nominal sequences only.
The testing subset is identical for both versions.
Furthermore, which model from the `model_garden` folder is trained can be set using the `MODEL_NAME` variable.

### 4. Inference
Before evaluating an arbitrary trained model, it needs to be run on the validation and test subsets, which is done with the `path/3_inference.py` script.
Again, `AD_MODE` and `MODEL_NAME` carry the same function as in the scripts preceding it.

### 5. Evaluation
With the inference outputs, the results can now be evaluated using the `path/4_evaluation.py` script.
Again, `AD_MODE` and `MODEL_NAME` carry the same function as in the scripts preceding it.

### TSADIS
Note that TSADIS can only be run in a different virtual environment due to its `Python 3.9` requirement.
The original paper provides no file declaring the dependencies required; hence, this repository provides an unofficial `requirements.txt` file.

The working script is `path/tsadis/tsadis.py`, which is very similar to the `path/3_inference.py` and `path/4_evaluation.py` scripts for the other approaches.
TSADIS is non-parametric and hence does not require a training procedure.
Functions are located in the `path/tsadis/helper_functions.py` script, which are accessed by `path/tsadis/tsadis.py`.

### Generalisation Study
The generalisation study requires a slightly different setup, since the data is split differently to the other experiments.
The procedure is almost identical to the one outlined above, and the relevant scripts can be found in the `path/generalisation` folder.

## Real-world Dataset
Unfortunately, this proprietary dataset is subject to strict confidentiality levels and therefore cannot be published.

The procedure is almost identical to the one outlined above, and the relevant scripts can be found in the `prop` folder.