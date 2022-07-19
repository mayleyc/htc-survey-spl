# Ticket Automation survey with Applications

Code used in paper: 

Zangari A, Marcuzzo M, Schiavinato M, Gasparetto A, Albarelli A. (2022) Ticket Automation: an Insight into Current Research with Applications to Multi-level Classification
      Scenarios. (In review)

## Datasets download

Instructions on how to download the The Linux bugs and Financial datasets.

1. Download them from [here](https://drive.google.com/file/d/119JPHB8iizeMI2C69iN2VqpXmARoyUIk/view?usp=sharing);
2. Unzip in the `data/` folder.

Note:

- the Financial dataset can also be downloaded from its [original source on Kaggle](https://www.kaggle.com/datasets/venkatasubramanian/automatic-ticket-classification);
- the script to scrape the Linux bugs dataset has been adapted from a [previous work](https://github.com/Forethought-Technologies/ieee-dsmp-2018-paper).

## Create necessary environments

Three separate environments are needed for PyTorch models, the SVM and DeepTriage that is implemented in Keras. Experiments are run on an Nvidia GeForce RTX 2080 Ti with CUDA 10.1
and Python 3.10.

1. PyTorch: `conda env create --file=env-torch.yml`;
2. SVM: `conda env create --file=env-svm.yml`;
3. Keras: `conda env create --file=env-keras.yml`.

Then activate the environment with `conda activate {NAME}` with NAME=`ticket-class-torch`, `ticket-class-svm`or `ticket-class-keras`.

## Run algorithms

All algorithms will report results in a `csv` file that will be created inside the folder of the main script.

### BERT models

Instructions to run BERT-based models.

1. Set desired parameters in `src/models/hierarchical_labeling/training_bert4c.py`. These include:
    - `dataset_global` for choosing the dataset;
    - `mode` to report metrics on training splits or validation splits;
    - `method` to select between 4 methods:
        - `BERT4C`: standard BERT LM with linear classification head on top (set params in `bert_config.yml`);
        - `ML-BERT`: Multi-Level model (ML-LM) (set params in `mllm_config.yml`);
        - `SUPP-BERT`: The supported classifier (SupportedLM) (set params in `supplm_config.yml`);
        - `DH-BERT`: The DoubleHeadLM classifier (set params in `dhlm_config.yml`).
2. Set hyperparameters in the yml configuration files for each model.
3. Run the script.

To run multiple models, have a look at our testing scripts. For instance `ensemble_tests.py` was used to test all the multilevel models.

### XLNet

1. Set up parameters in `src/models/xlnet/train_xlnet.py` and in the config scripts in the `src/models/xlnet/config/` folder;
2. Run `src/models/xlnet/train_xlnet.py`.

### DeepTriage

You need the Keras environment set up.

1. Define all hyperparameters in the yml files in `src/models/deep_triage/configs`;
2. Run `src/models/deep_triage/train_deeptriage.py`.

### FastText (from TicketTagger)

1. Set parameters in `src/models/ticket_tagger/config.yml`;
    - `autoTuneDuration` must be set in seconds;
    - Optionally specify the path to FastText pretrained vectors in `PRE_TRAINED_VECTORS` (we used [these](https://fasttext.cc/docs/en/crawl-vectors.html#models)).
2. Run `src/models/ticket_tagger/benchmark.py`.

### SVM

You need the SVM environment set up.

1. Set hyperparameters in `src/models/tfidf/configs/traditional_config.yml`. Many fields require a list of values that will be tried in a grid-search.
2. Run `src/models/tfidf/tfidf-svm.py`;

## Repository structure

- `src/dataset_tools` contains the script used to scrape the bugzilla repository as well as the utilities to load the datasets and create the barplots;
- `src/analysis` contains scripts to analyze the distribution of values in BERT embeddings;
- `src/model_evaluation` defines functions used to compute metrics;
- `src/models` contains the implementation of all the models we tested;
- `src/utils` defines some preprocessing functions and utilities to train PyTorch models.

## Citing this work

If you find this code or the provided data useful in your research, please consider citing:

```
   To be added after publication.
```
