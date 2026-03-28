This section presents how the results presented in Chapter 6 can be reproduced. 
The corresponding scripts are all located in the `active_learning` folder.

### 1. Parsing
First, download the PATH dataset, which is hosted on Zenodo and can be downloaded [here](https://zenodo.org/records/13255120).
Once downloaded, unzip either or both `0_simulation` and `1_postsim` folders to `data/path`.

If you only downloaded `0_simulation`, you need to parse it first using the `0_postsim.py` script.
The output will be identical to the contents of the `1_postsim` folder.

### 2. Preprocessing
Then, preprocess the contents of the `1_postsim` folder using `1_data.py`, which includes downsampling, standardisation, and windowing.

### 3. Training
Once preprocessed, the model training for different experiments can be run with the `2_training.py` script.
Like in `1_data.py`, the `AD_MODE` variable denotes whether the **unsupervised** or the **semi-supervised** version of the PATH dataset is used.
The difference between them is that the training and validation subsets of the **unsupervised** version contain unlabelled anomalous sequences, whereas the **semi-supervised** version contains nominal sequences only.
The testing subset is identical for both versions.
Furthermore, which model from the `model_garden` folder is trained can be set using the `MODEL_NAME` variable.

### 4. Inference
Before evaluating an arbitrary trained model, it needs to be run on the validation and test subsets, which is done with the `3_inference.py` script.
Again, `AD_MODE` and `MODEL_NAME` carry the same function as in the scripts preceding it.

### 5. Evaluation
Clearly, the procedure until now is very similar to the one to reproduce the results for Chapter 5.
Each query strategy has its own evaluation script.
The results for the unsupervised threshold and hypothetical best threshold baselines are provided by the `4_evaluation_unsupervised.py` and `4_evaluation_best.py` scripts, respectively.

Likewise, the results for the random-based, top-based, uncertainty-based and dissimilarity-based query strategies are provided by the `4_evaluation_random.py`, `4_evaluation_top.py`, `4_evaluation_uncertain.py` and `4_evaluation_ds.py` scripts, respectively.
Where applicable, the query budget and mislabelling probability can be changed using the `BUDGET` and `MISLABEL_PROB` variables.