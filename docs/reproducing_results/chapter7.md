This section presents how the results presented in Chapter 7 can be reproduced. 
The corresponding scripts are all located in the `pdm` folder.

### 1. Parsing
Unfortunately, this proprietary dataset is subject to strict confidentiality levels and therefore cannot be published.

The sampling frequency that the dataset is downsampled to is obtained the same way as for the PATH dataset in Chapter 4.
The corresponding code is found in the `pdm_downsampling.py` script.

The results were obtained as follows.
First, the raw MF4 files are parsed using the `0_parse.py` script.

### 2. Preprocessing
Then, preprocess the contents of the `1_parsed` folder using `1_data.py`, which includes downsampling, standardisation, and windowing.

### 3. Training
Once preprocessed, the multi-model TeVAE model training can be run with the `2_training.py` script.

### 4. Inference
Before evaluating the trained multi-model TeVAE model, it needs to be run on the validation and test subsets, which is done with the `3_inference.py` script.
Again, `AD_MODE` and `MODEL_NAME` carry the same function as in the scripts preceding it.

### 5. Evaluation
With the inference outputs, the results can now be evaluated using the `4_evaluation.py` script.
Again, `AD_MODE` and `MODEL_NAME` carry the same function as in the scripts preceding it.