# Hierarchical Text Classification survey
Implementations of HTC methods, used in paper:

- Marcuzzo M, Zangari A, Michele S, Matteo R, Gasparetto A, Albarelli A. (2022)
Hierarchical Text Classification: a review of current research. Submitted to Pattern Recognition.

> **NOTE**: set your working directory to `htc-survey-22` for imports to work properly.

## Installation

This procedure has been tested on Ubuntu 20.04.3 LTS with Python 3.8.11. Experiments were run on Nvidia GeForce RTX 2080
Ti with CUDA 10.1.

Install using conda:

```
conda env create -f environment.yml
```

Activate environment:

```
conda activate htc-survey
```

## Reproducing results

All source code for models are implemented in `src/models/{MODEL_NAME}`.
Test scripts are available in `src/training_scripts/...`.

Available models:
- BERT based multilabel classifier (+ versions with CHAMP and MATCH loss);
- SVM, multiclass and multilabel;
- XMLCNN, multilabel (+ versions with CHAMP and MATCH loss);
- HiAGM;
- MATCH (runs through a shell script utilizing the author's script, modified for new datasets).

## Dataset generation

Datasets should be in the `data` folder, organized with the following structure (the hierarchy files are in the `data/taxonomies` folder):
```
data/
├── BGC/            
│   ├── bgc_tax.txt
│   ├── BlurbGenreCollection_EN_test.jsonl
│   └── BlurbGenreCollection_EN_train.jsonl
├── RCV1v2/            
│   ├── rcv1_tax.txt
│   ├── test.jsonl
│   └── train.jsonl  
├── Amazon/          
│   ├── amazon_tax.txt
│   └── samples.jsonl
├── WebOfScience/            
│   ├── wos_tax.txt
│   └── samples.jsonl
├── Bugs/            
│   ├── bugs_tax.txt
│   └── all_linux_bugs.csv.gz
...                
```

We could not share our splits for RCV1-v2 due to its license. It is therefore necessary to acquire it from NIST as instructed in the [official page](https://trec.nist.gov/data/reuters/reuters.html).
We'll share instructions on how to generate our splits and use it with our models.

The BGC dataset is downloadable from the [official repository](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/blurb-genre-collection.html) with the "official" training and testing splits.
We'll share instructions on how to produce the JSONL files used in this work.

In order to ease comparison in future works, a training and testing split for the Amazon/WOS/Bugs datasets have been published on Zenodo at this [page](https://doi.org/10.5281/zenodo.7319518).

For the Amazon, WebOfScience and Bugs datasets the JSONL/CSV files can be obtained by merging the training and testing splits shared on Zenodo.

## Download embeddings

Some methods, like XML-CNN, require pretrained embeddings. These may be downloaded
from the following sources, which should then be placed in `data/embeddings/`:

- GloVe (en): https://nlp.stanford.edu/projects/glove


## Authors and acknowledgment



- **Matteo Marcuzzo**, Department of Environmental Sciences, Informatics and Statistics, Ca' Foscari University, Venice,
  Italy
- **Alessandro Zangari**, Department of Environmental Sciences, Informatics and Statistics, Ca' Foscari University, Venice,
  Italy
- **Michele Schiavinato**, Department of Environmental Sciences, Informatics and Statistics, Ca' Foscari University, Venice,
  Italy
- **Matteo Rizzo**, Department of Environmental Sciences, Informatics and Statistics, Ca' Foscari University, Venice,
  Italy
- **Andrea Gasparetto**, Department of Management, Ca' Foscari University, Venice, Italy
- **Andrea Albarelli**, Department of Environmental Sciences, Informatics and Statistics, Ca' Foscari University, Venice,
  Italy

## Citing this work
 (To add when published)