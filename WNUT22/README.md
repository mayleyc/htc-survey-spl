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

1. Download the compressed dataset from this [Google Drive folder](https://drive.google.com/file/d/1rFs7CcjV9zr8OxVQqmm1dy9g8iJRVgMK/view?usp=share_link) and move them to the `data`
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
@inproceedings{marcuzzo-etal-2022-multi,
   title = "A multi-level approach for hierarchical Ticket Classification",
   author = "Marcuzzo, Matteo  and
   Zangari, Alessandro  and
   Schiavinato, Michele  and
   Giudice, Lorenzo  and
   Gasparetto, Andrea  and
   Albarelli, Andrea",
   booktitle = "Proceedings of the Eighth Workshop on Noisy User-generated Text (W-NUT 2022)",
   month = oct,
   year = "2022",
   address = "Gyeongju, Republic of Korea",
   publisher = "Association for Computational Linguistics",
   url = "https://aclanthology.org/2022.wnut-1.22",
   pages = "201--214",
   abstract = "The automatic categorization of support tickets is a fundamental tool for modern businesses. Such requests are most commonly composed of concise textual descriptions that are noisy and filled with technical jargon. In this paper, we test the effectiveness of pre-trained LMs for the classification of issues related to software bugs. First, we test several strategies to produce single, ticket-wise representations starting from their BERT-generated word embeddings. Then, we showcase a simple yet effective way to build a multi-level classifier for the categorization of documents with two hierarchically dependent labels. We experiment on a public bugs dataset and compare our results with standard BERT-based and traditional SVM classifiers. Our findings suggest that both embedding strategies and hierarchical label dependencies considerably impact classification accuracy.",
}
```
