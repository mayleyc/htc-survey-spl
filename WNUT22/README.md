# W-NUT 2022 conference submission

Code repository for the article: `A multi-level approach for hierarchical Ticket Classification` accepted at [W-NUT 2022](https://noisy-text.github.io/2022).

## Installation

This procedure has been tested on Ubuntu 20.04.3 LTS with Python 3.10.4. Experiments were run on Nvidia GeForce RTX 2080
Ti with CUDA 11.

Install using conda:

```
conda env create -f environment.yml
```

Activate environment:

```
conda activate tc-conference
```

## Training our models

1. Download the compressed dataset from this [Google Drive folder](https://drive.google.com/file/d/1GJ2B7uv2Z5QcbeVzPChFQ1iG8swV-z7b/view?usp=sharing) and move them to the `data`
   folder;
2. Run the desired model from the `src/training_scripts` folder.
    - Multilevel models require prior training of the supporting models.

### Examples

#### Train BERT flat (base)

1. Setup hyperparameters in `configs/bert/bert_l1_config.yml`;
2. Run script `src/training_scripts/bert_clf/train_bert_flat.py`.

#### Train ML-BERT (base)

1. This requires that BERT-base flat models have been trained and dumps are available;
2. Setup hyperparameters in `configs/multi_level/multilevel_config.yml`;
    - Set up path to T1 and T2 model dumps by setting the `MODEL_L1` and `MODEL_L2` fields;
3. Run script `src/training_scripts/multilevel_models/train_multilevel.py`.

## Structure

```
WNUT22
└───configs
│   └───[model_name]
│   └───[...]
└───data
│   └───[dataset_name]
│   └───[...]
└───out
│   └───[model_name]
│   └───[...]
└───src
    └───dataset_tools
    │   │   └───classes
    │   │   └───data_preparation
    │   │   └───other
    └───models
    │   └───[model_name]
    │   └───[...]
    └───training_scripts
    │   └───[model_name]
    │   └───[...]
    └───utils
        └───text_utilities
        └───torch_train_eval
            └───[training utilities]
            └───[...]
```

## Citing

```
   To be added after publication
```